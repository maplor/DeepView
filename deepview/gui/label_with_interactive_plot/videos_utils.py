import os
import sqlite3
import cv2
import hashlib
import tempfile
import csv
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple, Dict, List, Union, Any



DB_PATH = os.path.join(os.path.dirname(__file__), "videos.db")

# ----------------- 基础工具函数 -----------------

def get_conn(dp_path=DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(dp_path)

def compute_file_hash(path: str, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def probe_video_metadata(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (frames / fps) if fps > 0 else 0.0
    return {"duration": duration, "width": width, "height": height}

def _normpath(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))

# ----------------- 时间转换工具 -----------------

def _parse_iso8601_to_unixtime(iso_str: str) -> Optional[float]:
    if not iso_str:
        return None
    s = iso_str.strip()
    try:
        # 支持形如 2018-08-28T06:00:00.000Z
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s2)
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        # 兜底尝试常见格式
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
    return None

def _format_unixtime_to_iso8601(ts: Union[int, float]) -> str:
    # 统一输出毫秒精度 + Z
    dt = datetime.utcfromtimestamp(float(ts))
    # timespec='milliseconds' 仅 3.8+，此处手动格式化到毫秒
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ms = int(round((float(ts) - int(ts)) * 1000.0))
    return f"{iso}.{ms:03d}Z"

# ----------------- 初始化与标签 -----------------

def init_db(db_path: str = DB_PATH):
    """
    初始化数据库及表结构。
    数据库字段说明：
    videos_path:
        id: 主键
        title: 视频标题
        path: 视频文件路径
        file_hash: 视频文件哈希
        duration: 视频时长，单位秒
        width: 视频宽度
        height: 视频高度
        start_time: 视频开始时间（ISO 8601 格式）
        start_unixtime: 视频开始时间（Unix 时间戳）
        created_at: 记录创建时间（ISO 8601 格式）
        file_size: 文件大小（字节）
        mtime: 文件最后修改时间（Unix 时间戳）
    tags:
        id: 主键
        name: 标签名称
    video_tags:
        video_id: 视频 ID，外键关联 videos_path(id)
        tag_id: 标签 ID，外键关联 tags(id)
    复合主键(video_id, tag_id)确保同一视频与标签的唯一关联。
    级联删除确保当视频或标签被删除时，相关联的记录也会被自动删除，保持数据一致性。


    """
    with get_conn(db_path) as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS videos_path (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                path TEXT UNIQUE,
                file_hash TEXT UNIQUE,
                duration REAL,
                width INTEGER,
                height INTEGER,
                start_time TEXT,
                stop_time TEXT,
                start_unixtime REAL,
                created_at TEXT,
                file_size INTEGER,
                mtime REAL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS video_tags (
                video_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY(video_id, tag_id),
                FOREIGN KEY(video_id) REFERENCES videos_path(id) ON DELETE CASCADE,
                FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
            """
        )
        # 兼容老库：如缺少 start_unixtime 列则补充
        try:
            c.execute("PRAGMA table_info(videos_path)")
            cols = {r[1] for r in c.fetchall()}
            if "start_unixtime" not in cols:
                c.execute("ALTER TABLE videos_path ADD COLUMN start_unixtime REAL")
            if "stop_time" not in cols:
                c.execute("ALTER TABLE videos_path ADD COLUMN stop_time TEXT")
            # 回填 stop_time：仅当存在开始时间与时长但 stop_time 为空
            try:
                backfill_rows = c.execute(
                    "SELECT id, start_unixtime, duration FROM videos_path WHERE stop_time IS NULL AND start_unixtime IS NOT NULL AND duration IS NOT NULL AND duration > 0"
                ).fetchall()
                for rid, s_uni, dur in backfill_rows:
                    try:
                        stop_iso = _format_unixtime_to_iso8601(float(s_uni) + float(dur))
                        c.execute("UPDATE videos_path SET stop_time=? WHERE id=?", (stop_iso, rid))
                    except Exception:
                        pass
                if backfill_rows:
                    print(f"已回填 stop_time 条数: {len(backfill_rows)}")
            except Exception:
                pass
        except Exception:
            pass
    print(f"数据库初始化完毕: {DB_PATH}")

def ensure_tag(conn, tag_name: str):
    tag_name = tag_name.strip()
    if not tag_name:
        return None
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (tag_name,))
    conn.commit()
    cur.execute("SELECT id FROM tags WHERE name=?", (tag_name,))
    row = cur.fetchone()
    return row[0] if row else None

# ----------------- 添加/查询/播放 -----------------

def add_video_path(
    path: str,
    title: Optional[str] = None,
    tags: Optional[str] = None,
    start_time: Optional[str] = None,
    start_unixtime: Optional[Union[int, float]] = None,
    db_path: str = DB_PATH
):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None
    file_hash = compute_file_hash(path)
    meta = probe_video_metadata(path)
    if meta is None:
        print(f"无法读取视频元数据: {path}")
        return None
    fsize = os.path.getsize(path)
    fmtime = os.path.getmtime(path)

    # 互补时间字段
    st_iso = start_time
    st_uni = float(start_unixtime) if start_unixtime is not None else None
    if st_iso and st_uni is None:
        parsed = _parse_iso8601_to_unixtime(st_iso)
        if parsed is not None:
            st_uni = parsed
    elif st_uni is not None and (not st_iso):
        st_iso = _format_unixtime_to_iso8601(st_uni)

    # 计算 stop_time （若有开始时间与时长）
    stop_iso: Optional[str] = None
    dur = meta.get("duration") if meta else None
    if st_uni is not None and dur is not None and dur > 0:
        stop_iso = _format_unixtime_to_iso8601(st_uni + float(dur))

    with get_conn(db_path) as conn:
        cur = conn.cursor()
        # 去重：根据 hash 检查
        cur.execute("SELECT id FROM videos_path WHERE file_hash=?", (file_hash,))
        ex = cur.fetchone()
        if ex:
            print(f"重复视频，已存在 ID={ex[0]}，跳过添加")
            return ex[0]

        cur.execute(
            """INSERT INTO videos_path(title, path, file_hash, duration, width, height, start_time, stop_time, start_unixtime, created_at, file_size, mtime)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                title or os.path.basename(path),
                path,
                file_hash,
                meta["duration"],
                meta["width"],
                meta["height"],
                st_iso if st_iso else None,
                stop_iso if stop_iso else None,
                st_uni if st_uni is not None else None,
                datetime.utcnow().isoformat(),
                fsize,
                fmtime,
            ),
        )
        video_id = cur.lastrowid
        # 处理标签
        if tags:
            for t in tags.split(","):
                tid = ensure_tag(conn, t)
                if tid:
                    cur.execute("INSERT OR IGNORE INTO video_tags(video_id, tag_id) VALUES(?,?)", (video_id, tid))
        conn.commit()
    print(
        f"已添加: {path}  id={video_id}  duration={meta['duration']:.2f}s  {meta['width']}x{meta['height']}  hash={file_hash}"
    )
    return video_id

def get_video_path_by_id(vid: int, db_path: str = DB_PATH) -> Optional[str]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT path FROM videos_path WHERE id=?", (vid,)).fetchone()
    return row[0] if row else None

def play_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    print(
        f"播放: {video_path}  FPS={cap.get(cv2.CAP_PROP_FPS):.2f} Size=({int(cap.get(3))}x{int(cap.get(4))})"
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------- 校验与扫描 -----------------

def verify_video(vid: int, update: bool = False, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT id, path, file_hash, file_size, mtime FROM videos_path WHERE id=?", (vid,)
        ).fetchone()
        if not row:
            print(f"ID={vid} 不存在")
            return
        _, path, old_hash, old_size, old_mtime = row
        if not os.path.exists(path):
            print(f"[缺失] ID={vid} 路径不存在: {path}")
            return

        now_size = os.path.getsize(path)
        now_mtime = os.path.getmtime(path)

        # 快速路径：若大小与mtime均未变（允许mtime 1秒内误差），且已有hash，则视为一致
        if old_hash and old_size is not None and old_mtime is not None:
            if (now_size == old_size) and (abs((now_mtime or 0) - (old_mtime or 0)) < 1.0):
                print(f"[OK-快速] ID={vid} 大小/时间未变，跳过哈希")
                return

        new_hash = compute_file_hash(path)
        if new_hash == old_hash:
            msg = f"[OK] ID={vid} 哈希一致"
            if update:
                meta = probe_video_metadata(path)
                cur.execute(
                    "UPDATE videos_path SET file_size=?, mtime=?, duration=?, width=?, height=? WHERE id=?",
                    (
                        now_size,
                        now_mtime,
                        meta["duration"] if meta else None,
                        meta["width"] if meta else None,
                        meta["height"] if meta else None,
                        vid,
                    ),
                )
                conn.commit()
                msg += "（已刷新元信息）"
            print(msg)
            return

        dup = cur.execute(
            "SELECT id, path FROM videos_path WHERE file_hash=? AND id<>?", (new_hash, vid)
        ).fetchone()
        if dup:
            print(
                f"[冲突] ID={vid} 新哈希与已存在记录 ID={dup[0]} 相同（路径: {dup[1]}）"
            )
            if update:
                print("已检测到重复内容，出于安全不自动更新。请手动处理去重或合并记录。")
            return

        if update:
            meta = probe_video_metadata(path)
            cur.execute(
                "UPDATE videos_path SET file_hash=?, file_size=?, mtime=?, duration=?, width=?, height=? WHERE id=?",
                (
                    new_hash,
                    now_size,
                    now_mtime,
                    meta["duration"] if meta else None,
                    meta["width"] if meta else None,
                    meta["height"] if meta else None,
                    vid,
                ),
            )
            conn.commit()
            print(f"[已更新] ID={vid} 哈希与元数据已刷新")
        else:
            print(f"[变更] ID={vid} 哈希不一致（未更新，传 update=True 可写回）")

def _is_video_ext(p: str, exts: Optional[Iterable[str]]) -> bool:
    if not exts:
        return True
    e = os.path.splitext(p)[1].lower()
    return e in {x.lower() if x.startswith(".") else f".{x.lower()}" for x in exts}

def scan_and_fix(root: str, exts: Optional[List[str]], update: bool, ingest: bool, dry_run: bool, db_path: str = DB_PATH):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        return

    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, path, file_hash FROM videos_path WHERE file_hash IS NOT NULL"
        )
        by_hash: Dict[str, List[Tuple[int, str]]] = {}
        for rid, rpath, rhash in cur.fetchall():
            if rhash:
                by_hash.setdefault(rhash, []).append((rid, rpath))

        seen_hashes = set()
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                fpath = os.path.join(dirpath, name)
                if not _is_video_ext(fpath, exts):
                    continue
                try:
                    size = os.path.getsize(fpath)
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    continue
                h = compute_file_hash(fpath)
                seen_hashes.add(h)
                if h in by_hash:
                    for vid, old_path in by_hash[h]:
                        if _normpath(old_path) != _normpath(fpath):
                            print(f"[移动识别] ID={vid}\n  旧: {old_path}\n  新: {fpath}")
                            if update and not dry_run:
                                cur.execute(
                                    "UPDATE videos_path SET path=?, file_size=?, mtime=? WHERE id=?",
                                    (os.path.abspath(fpath), size, mtime, vid),
                                )
                    if update and not dry_run:
                        conn.commit()
                else:
                    print(f"[新文件] {fpath}")
                    if ingest and update and not dry_run:
                        add_video_path(fpath, title=os.path.basename(fpath))

        cur.execute("SELECT id, path, file_hash FROM videos_path")
        missing = []
        for vid, p, h in cur.fetchall():
            if not os.path.exists(p):
                if not h or h not in seen_hashes:
                    missing.append((vid, p))
        for vid, p in missing:
            print(f"[仍缺失] ID={vid} 路径不存在：{p}")

# ----------------- 测试/演示包装 -----------------

# CSV 批量导入
CSV_EXAMPLE = (
    "path,title,tags,start_time,start_unixtime\n"
    "C:/videos/sample1.mp4,示例1,tagA;tagB,2018-08-28T06:00:00.000Z,\n"
    "C:/videos/sample2.avi,示例2,tagC,,1535436000\n"
)

def import_videos_from_csv(
    csv_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    dry_run: bool = False,
    columns: Optional[Dict[str, str]] = None,
    db_path: str = DB_PATH
):
    """从 CSV 导入视频。

    CSV 列：path,title,tags,start_time,start_unixtime
    - path: 必填，视频绝对或相对路径
    - title/tags: 可选，tags 可用逗号或分号分隔
    - start_time: ISO8601(如 2018-08-28T06:00:00.000Z)
    - start_unixtime: 秒(整数/浮点)

    参数 columns: 可选的列名映射，用于自定义 CSV 实际列名。
      例如：{
        'path': 'filepath',
        'title': 'name',
        'tags': 'labels',
        'start_time': 'start_iso',
        'start_unixtime': 'start_ts'
      }
      若未提供，则默认使用与逻辑名相同的列名。
    """
    total = added = skipped = errors = 0
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # 列名解析助手
        def col_name(key: str) -> str:
            return (columns.get(key) if columns else key) or key

        path_col = col_name("path")
        title_col = col_name("title")
        tags_col = col_name("tags")
        st_col = col_name("start_time")
        su_col = col_name("start_unixtime")

        for row in reader:
            total += 1
            try:
                path = (row.get(path_col) or "").strip()
                if not path:
                    skipped += 1
                    print(f"[CSV 跳过] 第{total}行: 缺少 path")
                    continue
                title_raw = row.get(title_col)
                title = title_raw.strip() if title_raw else None
                tags = (row.get(tags_col) or None)
                if tags:
                    # 兼容分号分隔
                    tags = tags.replace(";", ",")
                st_raw = row.get(st_col)
                st = st_raw.strip() if st_raw else None
                su_raw = (row.get(su_col) or "").strip()
                su: Optional[float] = None
                if su_raw:
                    try:
                        su = float(su_raw)
                    except ValueError:
                        su = None

                if dry_run:
                    print(f"[DRY-RUN] add {path} title={title} tags={tags} st={st} su={su}")
                else:
                    # 导入前先检查是否重复（按文件哈希）
                    if os.path.exists(path):
                        h = compute_file_hash(path)
                        with get_conn(db_path) as c2:
                            row = c2.execute(
                                "SELECT id FROM videos_path WHERE file_hash=?", (h,)
                            ).fetchone()
                        if row:
                            print(f"[CSV 跳过-重复] {path} 已存在 ID={row[0]}")
                            skipped += 1
                            continue
                    rid = add_video_path(path, title=title, tags=tags, start_time=st, start_unixtime=su, db_path=db_path)
                    if rid is None:
                        skipped += 1
                    else:
                        added += 1
            except Exception as e:
                errors += 1
                print(f"[CSV 错误] 第{total}行: {e}")
    print(f"CSV 导入完成: total={total} added={added} skipped={skipped} errors={errors}")
    return {"total": total, "added": added, "skipped": skipped, "errors": errors}





def parse_time_to_seconds(ts: str) -> float:
    '''将时间字符串解析为秒数
    Args:
        ts: 时间字符串，格式如 "HH:MM:SS.sss" 或 "MM:SS.sss" 或 "SS.sss"
    '''
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        parts = [p.strip() for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            raise ValueError("时间格式不正确，示例: 01:23:45.678 或 83.5")
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(ts)

def get_frame_at_time(video_path: str, t_sec: float, mode: str = "fast", backtrack: int = 15):
    '''根据时间戳获取视频帧
    Args:
        video_path: 视频文件路径
        t_sec: 目标时间（秒）
        mode: "fast" 或 "precise"
        backtrack: precise 模式下的回溯帧数
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # 仅占位，下面会用长度估计
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 限制 t_sec 在视频范围内（若能估算长度）
    video_len_sec = None
    if fps > 0 and frame_count > 0:
        video_len_sec = frame_count / fps
        t_sec = max(0.0, min(t_sec, max(0.0, video_len_sec - 1e-3)))

    if mode == "fast":
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        if not ok:
            raise RuntimeError("读取失败（fast）。")
        actual_time = actual_ms / 1000.0 if actual_ms > 0 else t_sec
        return frame, actual_time, None

    # precise 模式：按帧寻址 + 顺序解码到目标
    if fps <= 0:
        # 无法获取 FPS 时退化到 fast
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        if not ok:
            raise RuntimeError("读取失败（fallback fast）。")
        actual_time = actual_ms / 1000.0 if actual_ms > 0 else t_sec
        return frame, actual_time, None

    target_idx = int(round(t_sec * fps))
    if frame_count > 0:
        target_idx = max(0, min(target_idx, frame_count - 1))

    start_idx = max(0, target_idx - backtrack)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    cur_idx = start_idx
    chosen = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 估算当前时间（CFR 假设）
        cur_time = cur_idx / fps
        if cur_idx >= target_idx:
            chosen = (frame, cur_time, cur_idx)
            break
        cur_idx += 1

    cap.release()
    if chosen is None:
        raise RuntimeError("读取失败（precise）。")
    return chosen  # (frame, actual_time, frame_index)


# ----------------- 时间查找对应帧 -----------------
def find_frame_by_timestamp(
    ts: Union[int, float],
    tag: Optional[str] = None,
    mode: str = "fast",
    backtrack: int = 15,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """根据全局时间戳(ts, Unix秒)查找对应视频并抽取该时间点帧。

    逻辑:
    1. 在数据库中查找满足 start_unixtime <= ts < start_unixtime + duration 的视频记录。
    2. 如提供 tag，仅在该标签相关视频中查找（tags.name == tag）。
    3. 若匹配多个，取 mtime 最大(最新修改)的视频。
    4. 计算视频内部偏移 offset = ts - start_unixtime，调用 get_frame_at_time 获取帧。

    返回:
        dict 包含:
          video_id, video_path, timestamp(ts), video_start(start_unixtime),
          offset_in_video, actual_time_in_video, frame_index, tags(list[str]), frame(np.ndarray)
        若未找到或失败返回 None。
    """
    ts_f = float(ts)
    with get_conn(db_path) as conn:
        cur = conn.cursor()

        if tag:
            # 限定标签的视频
            rows = cur.execute(
                """
                SELECT vp.id, vp.path, vp.start_unixtime, vp.duration, vp.mtime
                FROM videos_path vp
                JOIN video_tags vt ON vt.video_id = vp.id
                JOIN tags t ON t.id = vt.tag_id
                WHERE t.name = ? AND vp.start_unixtime IS NOT NULL AND vp.duration IS NOT NULL
                AND ? >= vp.start_unixtime AND ? < vp.start_unixtime + vp.duration
                ORDER BY vp.mtime DESC
                """,
                (tag, ts_f, ts_f),
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT id, path, start_unixtime, duration, mtime
                FROM videos_path
                WHERE start_unixtime IS NOT NULL AND duration IS NOT NULL
                  AND ? >= start_unixtime AND ? < start_unixtime + duration
                ORDER BY mtime DESC
                """,
                (ts_f, ts_f),
            ).fetchall()

        if not rows:
            print(f"[查找帧] 未找到包含时间 {ts_f} 的视频记录 (tag={tag or 'ALL'})")
            return None

        # 选择最新 mtime 的第一条
        vid, vpath, vstart, vdur, vmtime = rows[0]
        if not os.path.exists(vpath):
            print(f"[查找帧] 匹配视频文件缺失: ID={vid} path={vpath}")
            return None

        offset = ts_f - float(vstart)
        if offset < 0 or offset >= float(vdur):  # 理论上不应该发生，因为SQL已约束
            print(f"[查找帧] 偏移越界 offset={offset:.3f} duration={vdur:.3f}")
            return None

        # 获取该视频的所有标签(用于调试/回传)
        tag_rows = cur.execute(
            """
            SELECT t.name FROM tags t
            JOIN video_tags vt ON vt.tag_id = t.id
            WHERE vt.video_id = ?
            ORDER BY t.name
            """,
            (vid,),
        ).fetchall()
        tag_list = [r[0] for r in tag_rows]

    try:
        frame, actual_time, frame_idx = get_frame_at_time(vpath, offset, mode=mode, backtrack=backtrack)
    except Exception as e:
        print(f"[查找帧] 抽帧失败: {e}")
        return None

    result = {
        "video_id": vid,
        "video_path": vpath,
        "timestamp": ts_f,
        "video_start": vstart,
        "offset_in_video": offset,
        "actual_time_in_video": actual_time,
        "frame_index": frame_idx,
        "tags": tag_list,
        "frame": frame,
    }
    print(
        f"[查找帧] 命中视频 ID={vid} path={vpath}\n  start={vstart:.3f} dur={vdur:.3f} ts={ts_f:.3f} offset={offset:.3f} actual={actual_time:.3f} frame_idx={frame_idx if frame_idx is not None else 'N/A'} tags={tag_list}"
    )
    return result




# 测试代码




# 示例视频路径
SAMPLE_VIDEO_PATH = r"C:\\Users\\user\\Desktop\\fast-ttt-2024-10-11\\videos\\9B36360\\PBOT0000.avi"
# 扫描的根目录
SAMPLE_SCAN_ROOT = os.path.dirname(SAMPLE_VIDEO_PATH)

def test_init_db():
    print("== test_init_db ==")
    init_db()

def test_add_video():
    print("== test_add_video ==")
    if os.path.exists(SAMPLE_VIDEO_PATH):
        vid = add_video_path(
            SAMPLE_VIDEO_PATH,
            tags="demo,test",
            start_time="2018-08-28T06:00:00.000Z",
            start_unixtime=None,
        )
        print(f"测试添加返回ID: {vid}")
    else:
        print(f"示例视频不存在，跳过添加: {SAMPLE_VIDEO_PATH}")

def test_import_csv():
    print("== test_import_csv ==")
    # 生成一个临时 CSV，若示例视频存在则用它演示
    rows = [
        {
            "path": SAMPLE_VIDEO_PATH,
            "title": "CSV-示例",
            "tags": "csv,import",
            "start_time": "",
            "start_unixtime": "1535436000",
        }
    ] if os.path.exists(SAMPLE_VIDEO_PATH) else []

    tmp = None
    tmp2 = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        with open(tmp.name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "title", "tags", "start_time", "start_unixtime"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        if rows:
            print(f"写入临时 CSV: {tmp.name}")
            import_videos_from_csv(tmp.name, dry_run=False)
            # 再写一个使用自定义列名的 CSV，并用列映射导入（dry-run演示）
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            with open(tmp2.name, "w", encoding="utf-8", newline="") as f2:
                writer2 = csv.DictWriter(
                    f2, fieldnames=["filepath", "name", "labels", "start_iso", "start_ts"]
                )
                writer2.writeheader()
                for r in rows:
                    writer2.writerow({
                        "filepath": r["path"],
                        "name": r["title"],
                        "labels": r["tags"],
                        "start_iso": r["start_time"],
                        "start_ts": r["start_unixtime"],
                    })
            print(f"写入临时 CSV(自定义列): {tmp2.name}")
            import_videos_from_csv(
                tmp2.name,
                dry_run=True,
                columns={
                    "path": "filepath",
                    "title": "name",
                    "tags": "labels",
                    "start_time": "start_iso",
                    "start_unixtime": "start_ts",
                },
            )
        else:
            print("无可用示例视频，改为仅打印 CSV 示例：\n" + CSV_EXAMPLE)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        if tmp2 is not None:
            try:
                os.unlink(tmp2.name)
            except OSError:
                pass

def test_import_csv_from_path(csv_path: str):
    print("== test_import_csv_from_path ==")
    if os.path.exists(csv_path):
        import_videos_from_csv(csv_path, dry_run=False)
    else:
        print(f"指定的 CSV 文件不存在: {csv_path}")

def test_play_video():
    print("== test_play_video ==")
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM videos_path ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        print("无记录可播放，先执行添加测试。")
        return
    path = get_video_path_by_id(row[0])
    if path and os.path.exists(path):
        print(f"播放最新记录 ID={row[0]} 路径={path} (按 q 退出窗口)")
        play_video(path)
    else:
        print("记录路径不存在，跳过播放。")

def test_verify(update: bool = False):
    print("== test_verify ==")
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM videos_path ORDER BY id").fetchall()]
    if not ids:
        print("无记录可校验。")
        return
    for vid in ids:
        verify_video(vid, update=update)

def test_scan():
    print("== test_scan ==")
    if not os.path.isdir(SAMPLE_SCAN_ROOT):
        print(f"扫描目录不存在，跳过: {SAMPLE_SCAN_ROOT}")
        return
    scan_and_fix(
        SAMPLE_SCAN_ROOT,
        exts=["mp4", "avi", "mov", "mkv", "flv", "wmv", "ts", "mpeg", "mpg"],
        update=False,  # 不写库
        ingest=False,
        dry_run=True,
    )

if __name__ == "__main__":
    test_init_db()
    # test_add_video()
    #  CSV 导入（示例路径不可用，打印 CSV 示例）
    # test_import_csv()
    test_import_csv_from_path("L:\\cc\\qh\\opencv\\test_csv.csv")
    # test_verify(update=False)
    # test_scan()
    # 播放会弹窗，最后执行
    # test_play_video()
    # 根据全局 Unix 时间戳查找帧
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, start_unixtime, duration FROM videos_path WHERE start_unixtime IS NOT NULL AND duration IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row:
        vid_latest, vstart_latest, vdur_latest = row
        # 选取视频开始后 5 秒（若不足 5 秒则取一半）作为演示时间戳
        offset_demo = 5.0 if vdur_latest > 6.0 else max(0.0, vdur_latest / 2.0)
        ts_demo = vstart_latest + offset_demo
        print(f"== 演示 find_frame_by_timestamp: video_id={vid_latest} ts_demo={ts_demo:.3f} (offset {offset_demo:.3f}) ==")
        result = find_frame_by_timestamp(ts_demo, tag=None, mode="fast")
        if result and result.get("frame") is not None:
            out_path = "L:\\cc\\qh\\opencv\\output_frame_lookup.jpg"
            if cv2.imwrite(out_path, result["frame"]):
                print(f"已保存抽帧到: {out_path}")
            else:
                print(f"保存抽帧失败: {out_path}")
        else:
            print("未成功抽取帧。")
    else:
        print("无带 start_unixtime 的视频记录，无法 find_frame_by_timestamp。")

    isotime = "2018-08-28T06:00:20.600Z"
    ts_lookup = _parse_iso8601_to_unixtime(isotime)

    if ts_lookup is None:
        print(f"无法解析时间字符串为时间戳: {isotime}")
    else:
        result = find_frame_by_timestamp(ts_lookup, tag="Omizunagidori", mode="precise", backtrack=10)
        # 直接显示结果
        if result and result.get("frame") is not None:
            cv2.imshow("Lookup Frame", result["frame"])
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

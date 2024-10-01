combobox_style = """
QWidget {
  background-color: none;
   color: #DFE1E2;
}
    QComboBox{
        background-color: #19232D;
    }
QComboBox QAbstractItemView {
  border: 1px solid #455364;
  border-radius: 0;
  background-color: #19232D;
  selection-background-color: #346792;
}

QComboBox QAbstractItemView:hover {
  background-color: #19232D;
  color: #DFE1E2;
}

QComboBox QAbstractItemView:selected {
  background: #346792;
  color: #455364;
}

QComboBox QAbstractItemView:alternate {
  background: #19232D;
}

QComboBox:disabled {
  background-color: #19232D;
  color: #788D9C;
}

QComboBox:hover {
  border: 1px solid #346792;
}

QComboBox:focus {
  border: 1px solid #1A72BB;
}

QComboBox:on {
  selection-background-color: #346792;
}

QComboBox::indicator {
  border: none;
  border-radius: 0;
  background-color: transparent;
  selection-background-color: transparent;
  color: transparent;
  selection-color: transparent;
  /* Needed to remove indicator - fix #132 */
}

QComboBox::indicator:alternate {
  background: #19232D;
}

QComboBox::item {
  /* Remove to fix #282, #285 and MR #288*/
  /*&:checked {
            font-weight: bold;
        }

        &:selected {
            border: 0px solid transparent;
        }
        */
}

QComboBox::item:alternate {
  background: #19232D;
}

QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 12px;
  border-left: 1px solid #455364;
}

QComboBox::down-arrow {
  image: url(":/qss_icons/dark/rc/arrow_down_disabled.png");
  height: 8px;
  width: 8px;
}

QComboBox::down-arrow:on, QComboBox::down-arrow:hover, QComboBox::down-arrow:focus {     
  image: url(":/qss_icons/dark/rc/arrow_down.png");
}



QPushButton {
  border: none; 
  color:#6D6D6D; 
font-size: 15px
}

QPushButton:disabled {
  background-color: #455364;
  color: #788D9C;
  border-radius: 4px;
  padding: 2px;
}

QPushButton:checked {
  background-color: #60798B;
  border-radius: 4px;
  padding: 2px;
  outline: none;
}

QPushButton:checked:disabled {
  background-color: #60798B;
  color: #788D9C;
  border-radius: 4px;
  padding: 2px;
  outline: none;
}

QPushButton:checked:selected {
  background: #60798B;
}

QPushButton:hover {
  background-color: #54687A;
  color: #DFE1E2;
}

QPushButton:pressed {
  background-color: #60798B;
}

QPushButton:selected {
  background: #60798B;
  color: #DFE1E2;
}

QPushButton::menu-indicator {
  subcontrol-origin: padding;
  subcontrol-position: bottom right;
  bottom: 4px;
}

QDialogButtonBox QPushButton {
  /* Issue #194 #248 - Special case of QPushButton inside dialogs, for better UI */
  min-width: 80px;
}                                  

"""

"""
QPushButton {
  background-color: #455364;
  color: #DFE1E2;
  border-radius: 4px;
  padding: 2px;
  outline: none;
  border: none;
}
"""
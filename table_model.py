from PySide2 import QtCore
import pandas as pd


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        pd.set_option('display.precision', 8)
        pd.set_option('display.float_format', lambda x: '%.8f' % x)
        self.table_data = data

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                self.dataChanged.emit(index, index)
                return str(self.table_data.iloc[index.row(), index.column()])
        return None

    def rowCount(self, index):
        return self.table_data.shape[0]

    def columnCount(self, index):
        return self.table_data.shape[1]

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.table_data.columns[col]
        return None

import json
import requests
import sys
import traceback
import pandas as pd

from PySide2.QtCore import *
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from table_model import TableModel

testing = 1
version = "1.1"
# region Misc. settings
''' Set portfolio path'''
if testing == 1:
    portfolio_path = "../portfolio2.json"
else:
    portfolio_path = "../portfolio.json"

''' Set API fetch interval'''
timer_interval = 5000

''' Set precision for prices'''
pd.set_option('display.precision', 8)
pd.options.display.float_format = '{:.8f}'.format
# endregion

# region Portfolio IO [Load/Save]
'''load portfolio file, returns pandas dataframe  '''


def load_portfolio():
    with open(portfolio_path, 'r') as f:
        return pd.read_json(f)


'''Save current coin info to json file  '''


def save_portfolio(data):
    with open(portfolio_path, 'w+') as f:
        json.dump(data, f)


# endregion


# region exchange APIs, coin lists


'''Binance API url '''
binance_api = 'https://api.binance.com/api/v3/ticker/price'

'''Kucoin API url'''
kucoin_api = "https://api.kucoin.com/api/v1/market/allTickers"

'''Coin List (updated after first pass)'''
binance_coins = []
kucoin_coins = []


# endregion

# region Helper Functions [PL]

def calculatePL(entry, current):
    return round(((current - entry) / abs(entry)) * 100, 2)

    # if pl > 0:
    #     return colors.prGreen(pl)
    # elif pl < 0:
    #     return colors.prRed(pl)
    # else:
    #     return colors.prWhite(pl)


def calculatePLSats(entry, current, amt):
    return round((amt * current) - (amt * entry), 5)


# endregion

# region Threading Stuff
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(object)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


# endregion

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        ''' Load our form.ui '''
        self.load_ui()

        ''' Load our widgets '''
        self.assign_widgets()

        # region Populate table

        '''Create model, pass data (has to be pandas dataframe)'''
        self.model = TableModel(load_portfolio())
        '''Set the model for table '''
        self.table.setModel(self.model)
        '''Set some table properties'''
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # self.table_setData()
        # self.table.setSortingEnabled(False)
        # self.table.resizeColumnsToContents()
        # endregion

        ''' Generate pool of threads '''
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # region  Timer to update table
        self.timer = QTimer()
        self.timer.setInterval(timer_interval)
        self.timer.timeout.connect(self.assign_worker_fn)
        self.timer.start()
        # endregion info

    # region Load Main UI/Widget Assignment/Save
    def load_ui(self):
        ''' loads ui file '''
        loader = QUiLoader()
        path = "form.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.w = loader.load(ui_file, self)
        self.w.show()
        self.w.setWindowTitle("Coin Tracker " + version)
        ui_file.close()

    def assign_widgets(self):
        self.table = self.w.table
        self.w.remove.clicked.connect(self.remove_rows)
        self.w.add.clicked.connect(self.add_row)

    def save(self):
        save_portfolio(self.model.table_data.to_dict(orient='records'))

    # endregion

    # region Table Manipulation [Remove Row/Add Dialog]
    def remove_rows(self):
        indexes = self.table.selectedIndexes()
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            index = indexes[0]
            # Remove the item and refresh.
            self.model.table_data.drop([index.row()], inplace=True)
            # print(self.model.table_data.head())
            # self.model.removeRow(index.row())
            self.model.layoutChanged.emit()
            self.table.clearSelection()
            # self.save()

    def add_row(self):

        ''' Loads dialog file  '''
        loader = QUiLoader()
        path = "dialog.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.dialog = loader.load(ui_file, self)
        self.dialog.setWindowTitle("Add Coin")
        self.dialog.show()

        ''' Completer works only after first iteration of refresh/query  '''
        complete_list = list(set(binance_coins + kucoin_coins))
        completer = QCompleter(complete_list)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.dialog.coin_name.setCompleter(completer)

        ''' Link buttons  '''
        self.dialog.add.clicked.connect(self.dialog_add_button)
        self.dialog.cancel.clicked.connect(self.close_dialog)

        ui_file.close()

    # endregion
    # region Dialog [Add Coin/Close]
    def dialog_add_button(self):
        coinname = self.dialog.coin_name.text()
        price = self.dialog.entry_price.text()
        exchange = self.dialog.exchange_select.currentText()
        amount = self.dialog.amount.currentText()

        if len(coinname) > 0 and float(price) > 0:
            ''' add field data to coindata df  '''
            coin_to_add = {"symbol": [coinname.upper()],
                           "exchange": [exchange],
                           "entry": [float(price)],
                           "price": [0.0],
                           "P/L %": [0.0],
                           "P/L": [0.0],
                           "amount": [0.0]
                           }

            data = pd.DataFrame.from_dict(coin_to_add)
            self.model.table_data = self.model.table_data.append(data, ignore_index=True, sort=False)
            self.model.layoutChanged.emit()
            # print(self.model.table_data.head())

            self.close_dialog()
        else:
            self.dialog.coin_name.setPlaceholderText("Please enter coin name")
            self.dialog.entry_price.setPlaceholderText("Please enter entry price")

    def close_dialog(self):
        # Change app settings so we can close dialog without closing the whole application
        # Also restart the timer so we can continue updating the data live
        app.setQuitOnLastWindowClosed(False)
        self.dialog.reject()
        app.setQuitOnLastWindowClosed(True)

    # def progress_fn(self, data):
    #     print(data)
    # endregion

    # region Executing Functions
    # def fetch_price(self, progress_callback):
    # Need to filter this data based on our coins
    # progress_callback.emit(data)

    def fetch_price(self, progress_callback):
        # print("executing")
        return requests.get(url=binance_api).content, requests.get(url=kucoin_api).json()

    def handle_output(self, data):
        global binance_coins, kucoin_coins

        (binance_data, kucoin_data) = data

        ''' Filters our data from fetch_price to exclude coins we dont have in the list.  '''
        binance_new_data = pd.read_json(binance_data)

        kucoin_new_data = pd.DataFrame.from_records(pd.DataFrame(kucoin_data['data'])['ticker'].values)
        kucoin_new_data['symbol'] = kucoin_new_data['symbol'].apply(lambda x: "".join(x.split("-"))).tolist()

        if len(binance_coins) == 0:
            binance_coins = binance_new_data['symbol'].tolist()
        if len(kucoin_coins) == 0:
            kucoin_coins = kucoin_new_data['symbol'].tolist()

        binance_cleaned = binance_new_data[
            binance_new_data['symbol'].isin(self.model.table_data['symbol'].tolist())].reset_index(drop=True)
        kucoin_cleaned = kucoin_new_data[
            kucoin_new_data['symbol'].isin(self.model.table_data['symbol'].tolist())].reset_index(drop=True)

        for row in self.model.table_data.iterrows():
            coin = row[1]['symbol']
            exchange = row[1]['exchange']
            if exchange.lower() == "binance":
                newprice = binance_cleaned.loc[binance_cleaned['symbol'] == coin]['price'].item()
            else:
                newprice = kucoin_cleaned.loc[kucoin_cleaned['symbol'] == coin]['averagePrice'].item()

            self.model.table_data.loc[self.model.table_data['symbol'] == coin, 'price'] = newprice

        self.model.table_data['entry'] = self.model.table_data.apply(
            lambda x: '%.2f' % float(x['entry']) if x['symbol'].upper().find("USD") != -1 else '%.8f' % float(
                x['entry']),
            axis=1).astype(str)
        #
        self.model.table_data['price'] = self.model.table_data.apply(
            lambda x: '%.2f' % float(x['price']) if x['symbol'].upper().find("USD") != -1 else '%.8f' % float(
                x['price']),
            axis=1).astype(str)

        self.model.table_data['P/L %'] = self.model.table_data.apply(
            lambda x: str(calculatePL(float(x['entry']), float(x['price']))) + "%",
            axis=1)

        self.model.table_data['P/L'] = self.model.table_data.apply(
            lambda x: str(calculatePLSats(float(x['entry']), float(x['price']), float(x['amount']))) + " ₿",
            axis=1)

        moneymoney = self.model.table_data['P/L'].apply(lambda x: float(x.split(" ")[0])).sum()
        palette = self.w.lcdNumber.palette()
        if moneymoney > 0:
            palette.setColor(palette.WindowText, QColor(0,128,0))
        elif moneymoney < 0:
            palette.setColor(palette.WindowText, QColor(255,0,0))

        self.w.lcdNumber.setPalette(palette)
        self.w.lcdNumber.display(moneymoney)

        self.model.layoutChanged.emit()

    def thread_complete(self):
        """When fetching/updating is complete, save"""
        self.save()

    def assign_worker_fn(self):
        # Pass the function to execute
        worker = Worker(self.fetch_price)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.handle_output)
        worker.signals.finished.connect(self.thread_complete)
        # worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)


# endregion


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    app.exec_()

import pandas as pd

SCATS_DATA_OCTOBER_2006_CSV_PATH = 'datasets/Scats Data October 2006.csv'
SCATS_DATA_OCTOBER_2006_XLS_PATH = 'datasets/Scats Data October 2006.xlsb'
SCATS_SITE_LISTING_SPREADSHEET_VICROADS_CSV_PATH = 'datasets/SCATSSiteListingSpreadsheet_VicRoads.csv'
SCATS_SITE_LISTING_SPREADSHEET_VICROADS_XLS_PATH = 'datasets/SCATSSiteListingSpreadsheet_VicRoads.xls'
TRAFFIC_COUNT_LOCATIONS_WITH_LONG_LAT_CSV_PATH = 'datasets/Traffic_Count_Locations_with_LONG_LAT.csv'


#  printing xlsb file
def PrintXLSB():
    with pd.ExcelFile(SCATS_DATA_OCTOBER_2006_XLS_PATH, engine='pyxlsb') as xls:
        workbook = xls.sheet_names
        print(workbook)

        df = pd.read_excel(xls, sheet_name=workbook[0])

    # returning the dataframe
    return df.head(1000).to_string()


# printing csv file
def PrintCSV():
    df = pd.read_csv(SCATS_DATA_OCTOBER_2006_CSV_PATH)
    # returning the dataframe
    return df.to_string()


print(PrintXLSB())

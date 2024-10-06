import pandas as pd


if __name__ == "__main__":
    data_2023 = pd.read_csv('data/load_values_23.csv', delimiter='\t')
    data_2023 = data_2023.query("CountryCode == 'GR'").reset_index()
    data_2023 = data_2023.drop(["CreateDate", "UpdateDate", "CountryCode", "index", 
                    "MeasureItem", "DateShort", "TimeFrom", "TimeTo",
                    "Cov_ratio", "Value_ScaleTo100"], axis=1)
    data_2023['DateUTC'] = pd.to_datetime(data_2023['DateUTC'], dayfirst=True)
    data_2023.iloc[2018, 0] = pd.Timestamp(2023, 3, 26, 2)

    data_2024 = pd.read_csv('data/load_values_24.csv')
    data_2024 = data_2024.query("CountryCode == 'GR'").reset_index()
    data_2024 = data_2024.drop(["CreateDate", "UpdateDate", "CountryCode", "index", 
                    "MeasureItem", "DateShort", "TimeFrom", "TimeTo",
                    "Cov_ratio", "Value_ScaleTo100"], axis=1)
    data_2024['DateUTC'] = pd.to_datetime(data_2024['DateUTC'], dayfirst=True)
    data_2024.loc[len(data_2024.index)] = [pd.Timestamp(2024, 3, 31, 2), data_2024['Value'].mean()]
    data_2024.loc[len(data_2024.index)] = [pd.Timestamp(2024, 3, 31, 3), data_2024['Value'].mean()]
    data_2024 = data_2024.sort_values('DateUTC')

    pd.concat([data_2023, data_2024]).to_csv("data/load_data_23_24.csv", index=False)
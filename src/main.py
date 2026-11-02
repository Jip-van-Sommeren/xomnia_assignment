import pandas as pd
from pydantic import BaseModel, NonNegativeInt, TypeAdapter
from datetime import datetime


class PlayerCount(BaseModel):
    app_id: int
    Time: datetime
    Playercount: NonNegativeInt | None


class GameInfo(BaseModel):
    appid: int
    type: str | None
    name: str
    releasedate: datetime | None
    freetoplay: bool | None


adapter = TypeAdapter(list[PlayerCount])


def validate_pc_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    records = data.astype(object).where(data.notna(), None).to_dict("records")

    rows = adapter.validate_python(records)

    result = pd.DataFrame(row.model_dump() for row in rows)
    result = result.astype({"app_id": "int64", "Playercount": "Int64"})
    result["Time"] = pd.to_datetime(result["Time"])
    return result


month_map = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def convert_date(date):
    if pd.isna(date):
        return None
    temporary_date = date.split("-")
    if len(temporary_date) != 3:
        month, year = temporary_date
        month = month_map[month]
        return f"01-{month}-{year}"
    day, month, year = temporary_date
    month = month_map[month]
    return f"{day}-{month}-{year}"


def validate_gi_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data["releasedate"] = data["releasedate"].apply(convert_date)
    print(data["releasedate"].head())

    data["releasedate"] = pd.to_datetime(
        data["releasedate"], format="%d-%m-%y", errors="raise"
    )

    records = data.astype(object).where(data.notna(), None).to_dict("records")

    rows = TypeAdapter(list[GameInfo]).validate_python(records)

    result = pd.DataFrame(row.model_dump() for row in rows)
    result = result.astype({"appid": "int64", "freetoplay": "bool"})

    return result


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a parsing error while reading the file.")
        return None


def main():
    file_path = (
        "data/applicationInformation.csv"  # Update this path to your CSV file
    )
    data = pd.concat(
        (
            validate_gi_dataframe(chunk)
            for chunk in pd.read_csv(
                file_path, chunksize=50_000, encoding="cp1252"
            )
        ),
        ignore_index=True,
    )
    # data["Playercount"] = data["Playercount"].fillna(0)
    # data["Time"] = pd.to_datetime(data["Time"])
    if data is not None:
        print("Data loaded successfully:")
        print(data[data["name"].isna()])
        print(data.head())  # Display the first few rows of the DataFrame
    else:
        print("Failed to load data.")


if __name__ == "__main__":
    main()

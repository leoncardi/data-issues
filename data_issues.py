import pandas as pd

class DataIssues:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self) -> str:
        return f'DataIssues(data={self.data.columns})'

    def get_data_types(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.dtypes, columns=['data_type']).T

    def count_missing_data_issues(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.isnull().sum(), columns=['missing_data']).T

    def calculate_outlier_count(self, column: str) -> tuple:
        """
        Calculate the number of upper and lower outliers using the Interquartile Range (IQR) method.
        
        Args:
            column (str): Name of the column to analyze.
        
        Returns:
            tuple: Number of lower outliers and upper outliers in the column.
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        iqr = Q3 - Q1

        lower_limit = Q1 - 1.5 * iqr
        upper_limit = Q3 + 1.5 * iqr

        lower_outliers = self.data[self.data[column] < lower_limit].shape[0]
        upper_outliers = self.data[self.data[column] > upper_limit].shape[0]

        return lower_outliers, upper_outliers

    def count_outliers(self) -> pd.DataFrame:
        """
        Get the number of outliers and the count of data points in all numeric columns of the DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with the number of outliers and data points for each column.
        """
        outlier_counts = {}
        data_points_counts = {}
        lower_outliers_counts = {}
        upper_outliers_counts = {}
        outlier_counts_percent = {}
        lower_outliers_percent = {}
        upper_outliers_percent = {}

        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                lower_outliers, upper_outliers = self.calculate_outlier_count(column)
                lower_outliers_counts[column] = lower_outliers
                upper_outliers_counts[column] = upper_outliers
                outlier_counts[column] = lower_outliers + upper_outliers
                data_points_counts[column] = self.data[column].count()
                if data_points_counts[column] > 0:
                    outlier_counts_percent[column] = round(outlier_counts[column] / data_points_counts[column], 3)
                    lower_outliers_percent[column] = round(lower_outliers / data_points_counts[column], 3)
                    upper_outliers_percent[column] = round(upper_outliers / data_points_counts[column], 3)
                else:
                    outlier_counts_percent[column] = 0
                    lower_outliers_percent[column] = 0
                    upper_outliers_percent[column] = 0
            else:
                lower_outliers_counts[column] = 0
                upper_outliers_counts[column] = 0
                outlier_counts[column] = 0
                data_points_counts[column] = self.data[column].count()
                outlier_counts_percent[column] = 0
                lower_outliers_percent[column] = 0
                upper_outliers_percent[column] = 0

        return pd.DataFrame({
            'data_points': data_points_counts,
            'outliers': outlier_counts,
            'upper_outliers': upper_outliers_counts,
            'lower_outliers': lower_outliers_counts,
            'ratio_outliers': outlier_counts_percent,
            'ratio_upper_outliers': upper_outliers_percent,
            'ratio_lower_outliers': lower_outliers_percent
        }).T

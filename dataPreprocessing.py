import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor


class DataPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def grab_col_names(self, cat_th=10, car_th=20):
        """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optional
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.

        """
        # cat_cols, cat_but_car
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if
                       self.dataframe[col].nunique() < cat_th and self.dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in self.dataframe.columns if
                       self.dataframe[col].nunique() > car_th and self.dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {self.dataframe.shape[0]}")
        print(f"Variables: {self.dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, num_cols, cat_but_car

    def cat_summary(self, col_name, plot=False):
        print(pd.DataFrame({col_name: self.dataframe[col_name].value_counts(),
                            "Ratio": 100 * self.dataframe[col_name].value_counts() / len(self.dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=self.dataframe[col_name], data=self.dataframe)
            plt.show()

    def num_summary(self, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(self.dataframe[numerical_col].describe(quantiles).T)

        if plot:
            self.dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()

    def target_summary_with_cat(self, target, categorical_col):
        print(categorical_col)
        print(pd.DataFrame({"TARGET_MEAN": self.dataframe.groupby(categorical_col)[target].mean(),
                            "Count": self.dataframe[categorical_col].value_counts(),
                            "Ratio": 100 * self.dataframe[categorical_col].value_counts() / len(self.dataframe)}),
              end="\n\n\n")

    def columnsName(self):
        columnsNames = list(self.data.columns)
        return columnsNames

    def high_correlated_cols(self, plot=False, corr_th=0.90):
        corr = self.dataframe.corr()
        cor_matrix = corr.abs()
        upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
        if plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set(rc={'figure.figsize': (15, 15)})
            sns.heatmap(corr, cmap="RdBu")
            plt.show()
        return drop_list

    def target_correlation_matrix(self, corr_th=0.5, target="Salary"):
        """
        Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
        :param dataframe:
        :param corr_th: eşik değeri
        :param target:  bağımlı değişken ismi
        :return:
        """
        corr = self.dataframe.corr()
        corr_th = corr_th
        try:
            filter = np.abs(corr[target]) > corr_th
            corr_features = corr.columns[filter].tolist()
            sns.clustermap(self.dataframe[corr_features].corr(), annot=True, fmt=".2f")
            plt.show()
            return corr_features
        except:
            print("Yüksek threshold değeri, corr_th değerinizi düşürün!")

    def removeMissingValue(self):
        print(self.data.isnull.sum())
        self.data = self.data.dropna(inplace=True)
        print("işlem sonrası veriseti")
        print(self.data.isnull.sum())
        return self.data

    def eksikVeriyeOrtalamaAtama(self):
        print(self.data.isnull.sum())
        self.data = self.data.fillna(self.data.mean()[:])
        print("işlem sonrası veriseti")
        print(self.data.isnull.sum())
        return self.data

    def outlier_thresholds(self, col_name, q1=0.25, q3=0.75):
        quartile1 = self.dataframe[col_name].quantile(q1)
        quartile3 = self.dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(self, variable, q1=0.25, q3=0.75):
        low_limit, up_limit = DataPreprocessing.outlier_thresholds(self.dataframe, variable, q1, q3)
        self.dataframe.loc[(self.dataframe[variable] < low_limit), variable] = low_limit
        self.dataframe.loc[(self.dataframe[variable] > up_limit), variable] = up_limit

    def check_outlier(self, col_name, q1=.25, q3=.75):
        low_limit, up_limit = DataPreprocessing.outlier_thresholds(self.dataframe, col_name, q1, q3)
        if self.dataframe[(self.dataframe[col_name] > up_limit) | (self.dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False

    def grab_outliers(self, col_name, index=False):
        low, up = DataPreprocessing.outlier_thresholds(self.dataframe, col_name)
        if self.dataframe[((self.dataframe[col_name] < low) | (self.dataframe[col_name] > up))].shape[0] > 10:
            print(self.dataframe[((self.dataframe[col_name] < low) | (self.dataframe[col_name] > up))].head())
        else:
            print(self.dataframe[((self.dataframe[col_name] < low) | (self.dataframe[col_name] > up))])

        if index:
            outlier_index = self.dataframe[((self.dataframe[col_name] < low) | (self.dataframe[col_name] > up))].index
            return outlier_index

    def remove_outlier(self, col_name):
        low_limit, up_limit =  DataPreprocessing.outlier_thresholds(self.dataframe, col_name)
        df_without_outliers = self.dataframe[~((self.dataframe[col_name] < low_limit) | (self.dataframe[col_name] > up_limit))]
        return df_without_outliers

    def missing_values_table(dataframe, na_name=False):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
        print(missing_df, end="\n")
        if na_name:
            return na_columns

    def missing_vs_target(dataframe, target, na_columns):
        temp_df = dataframe.copy()
        for col in na_columns:
            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
        na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
        for col in na_flags:
            print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                                "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

    def label_encoder(self, binary_col):
        labelencoder = LabelEncoder()
        self.dataframe[binary_col] = labelencoder.fit_transform(self.dataframe[binary_col])
        return self.dataframe

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    def rare_analyser(dataframe, target, cat_cols):
        for col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()

        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

        return temp_df

    # def kategorikDonustur(self):
    #     lbe = LabelEncoder()
    #     self.data["sex"]=lbe.fit_transform(self.data["sex"])
    #     return self.data
    #
    # def oneHotEncoding(self):
    #     self.data = pd.get_dummies(self.data, columns=["sex"], prefix=["sex"])
    #     return self.data

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
showWarningOnDirectExecution = False
st.set_page_config(
    page_title="Linear Regression",
    page_icon="chart_with_upwards_trend")

st.title(":red[Linear Regression Assumptions]")
st.subheader("*Just* *Add* your ***:rainbow[File]*** :file_folder:",divider='violet')
try:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([":blue[**Data**]", ":blue[**Linearity**]", ":blue[**Multicollinearity**]",":blue[**Normality**]",":blue[**Homoscedasticity**]",":blue[**Autocorrelation**]"])
    with tab1:
        spectra = st.file_uploader("upload file", type={"csv","xlsx"})
        if spectra is not None:
    # Determine file type and read accordingly
            if spectra.name.endswith(".xlsx"):
                spectra_df = pd.read_excel(spectra)
                st.write("Excel")
            elif spectra.name.endswith(".csv"):
                spectra_df = pd.read_csv(spectra)
                st.write("Csv")
            st.divider()
            st.write("**Data**")
            st.write(spectra_df.head(5))
            st.divider()
            col1, col2= st.columns(2)
            with col1:
                dep = st.radio(
                "**Select the dependent variable?**",spectra_df.columns[:])
            y=spectra_df[dep]
            with col2:
                st.markdown("**Dependent Variable:**")
                st.write(dep)
            st.divider()
            ##
            X=spectra_df.drop([dep],axis=1)
            ##
            options=st.multiselect("**Remove unwanted variable**",X.columns[:])
            st.caption("**Skip if you do not want to select anything.**")
            st.write(options)
            X=X.drop(columns=options)
            st.write(X.head())
            st.divider()
            ##
            st.markdown("**Independent Variables:**")
            st.write(X.columns[:])
            st.divider()
            ##

            # Using numeric values 
            numeric = X.select_dtypes(include='number')

            #Convert categorical variables
            dummy=st.multiselect("**Select Categorical Variable**",X.columns[:])
            st.caption("**Skip if you do not want to select anything.**")
            st.write(dummy)
            X=pd.get_dummies(X, columns=dummy,drop_first=True)
            for column in X.columns:
               if X[column].dtype == 'bool':
                    X[column] = X[column].astype(int)
            st.write(X.head())
            st.divider()
            ##
            if st.button(label="Confirm"):
                st.write(":large_green_circle: **Ready to Go** :large_green_circle:")
                model = LinearRegression()
                model.fit(X,y)
                predictions = model.predict(X)
                residual=y-predictions.reshape(-1)
    with tab2:
         fig1 = plt.figure(figsize=(10, 5))
         sns.residplot(data=spectra_df,x=predictions,y=y,lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
         st.subheader("**Residual Plot**",divider='blue')
         plt.xlabel("Fitted values")
         plt.ylabel("Residual")
         st.pyplot(fig1)
         st.caption("**If the red line is close to the dotted line it means there is linear relationship between the dependent variables and independent variables**")
    with tab3:
        if X.shape[1]>1:
            st.subheader("**Heatmap**",divider='blue')
            fig2=plt.figure(figsize=(8,6))
            sns.heatmap(numeric.corr(numeric_only = True),annot=True)
            st.pyplot(fig2)
            ##VIF
            st.subheader("**Variance Inflation Factor**",divider='blue')
            vif = []
            for i in range(numeric.shape[1]):
                vif.append(variance_inflation_factor(numeric, i))
            vif_1=pd.DataFrame({'vif': vif}, index=numeric.columns[:]).T
            st.table(vif_1)
            st.caption("**If VIF>5 there is a problem multicollinearity**")
        else:
            st.write("There is no problem of Multicollinearity")
    with tab4:
        # Create a QQ plot in Streamlit
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(residual, dist="norm", plot=ax,fit=False)

        # Customize the plot
        ax.set_title("Normal Q-Q Plot of Residuals", fontsize=16)
        ax.set_xlabel("Theoretical Quantiles", fontsize=14)
        ax.set_ylabel("Sample Quantiles", fontsize=14)
        ax.grid()

        # Add a 45-degree reference line
        min_val = min(residual)
        max_val = max(residual)
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x (45-degree line)')
        # ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
        ## KS Test
        ks=kstest(residual,'norm')
        st.subheader("**Kolmogorov Smirnov test**",divider='blue')
        st.write("p-value:",ks[1])
        if ks[1]<0.05:
            st.subheader("Residual is not normally distributed")
        else:
            st.subheader("Residual is normally distributed")
        # shapiro_stat, p_value = stats.shapiro(residual)
        # st.write("p-value:",p_value)
    with tab5:
        model_norm_residuals_abs_sqrt=np.sqrt(np.abs(residual))
        st.subheader("**Scale-Location**",divider='blue')
        fig4=plt.figure(figsize=(7,7))
        sns.regplot(data= spectra_df,x=predictions,y=model_norm_residuals_abs_sqrt,
                      scatter=True,lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plt.ylabel("Standarized residuals")
        plt.xlabel("Fitted value")
        st.pyplot(fig4)
        ##
        st.subheader("**Goldfeld-Quandt Test**",divider='blue')
        test = sms.het_goldfeldquandt(residual, X)
        name = ['F statistic', 'p-value']
        st.table(lzip(name, test))
        if test[1]<0.05:
            st.subheader("Data is heteroscedatic")
        else:
            st.subheader("Data is homoscedastic")
    with tab6:
        st.subheader("**Residual Vs Observation**",divider='blue')
        fig5= plt.figure(figsize=(15,6))
        plt.plot(residual)
        st.pyplot(fig5)
        st.subheader("**Durbin-Watson Test**",divider='blue')
        dw=durbin_watson(residual)
        st.markdown("**Durbin-Watson value:**")
        st.write(dw)
        if dw < 1.5 and dw > 0 :
            st.subheader("There is positive autocorrelation")
        elif dw == 2:
            st.subheader("There is no autocorrelation")
        elif dw >1.5 and dw < 2.5:
            st.subheader("There is slight autocorrelation")
        elif dw >2.5 and dw < 4:
            st.subheader("There is negative autocorrelation")
except:
        st.write("Please load a file to continue... / Click Confirm")

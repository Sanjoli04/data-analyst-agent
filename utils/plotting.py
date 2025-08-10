# utils/plotting.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from sklearn import linear_model

def create_plot(df, x_col, y_col, plot_type='scatter', regression=False, hue_col=None):
    """
    Creates a plot from a DataFrame and returns a base64-encoded image URI.
    """
    plt.figure(figsize=(10, 6))
    
    try:
        if plot_type == 'scatter':
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='viridis', legend='full' if hue_col else False)
            plt.title(f'Scatter Plot of {y_col} vs. {x_col}')
        
        if regression and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            X_reg = df[[x_col]].dropna()
            y_reg = df[y_col].loc[X_reg.index]
            if not X_reg.empty:
                reg_line = linear_model.LinearRegression().fit(X_reg, y_reg)
                plt.plot(X_reg, reg_line.predict(X_reg), color='red', linestyle='--', label='Regression Line')
    
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        if regression or hue_col:
            plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        plt.close()
        print(f"Error creating plot: {e}")
        return {"error": f"Failed to create plot: {e}"}

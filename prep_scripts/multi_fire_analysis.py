import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# LOAD DATA
df = pd.read_csv("/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/aois_and_fires_good.csv")

# Keep only rows with valid burn history and degradation metric
df_clean = df.dropna(subset=['num_fires', 'fire_age_last', 'number_of_edges']).copy()

# EXPLORATORY VISUALIZATION
def plot_edges_by_fire_frequency(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="num_fires", y="number_of_edges")
    sns.stripplot(data=df, x="num_fires", y="number_of_edges", 
                  color="black", alpha=0.4)
    plt.title("Degradation (Edges) vs. Fire Frequency")
    plt.show()


def plot_edges_vs_recency(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x="fire_age_last", y="number_of_edges", 
                    hue="num_fires", palette="viridis")
    plt.title("Degradation vs. Time Since Last Fire (colored by fire frequency)")
    plt.show()

# STANDARDIZATION FOR FAIR EFFECT SIZE COMPARISON
def standardize_predictors(df):
    scaler = StandardScaler()
    df[['num_fires_z', 'fire_age_last_z']] = scaler.fit_transform(
        df[['num_fires', 'fire_age_last']]
    )
    return df

df_clean = standardize_predictors(df_clean)


# LINEAR MODEL: MAIN EFFECTS ONLY
def linear_model_main_effects(df):
    model = ols("number_of_edges ~ num_fires_z + fire_age_last_z", data=df).fit()
    print("=== Main Effects Model ===")
    print(model.summary())
    return model

# LINEAR MODEL: INTERACTION EFFECT
def linear_model_with_interaction(df):
    model = ols("number_of_edges ~ num_fires_z * fire_age_last_z", data=df).fit()
    print("=== Interaction Model ===")
    print(model.summary())
    return model


# Fire frequency vs fire recency plot
def plot_stacked_fire_frequency_by_recency(df):
    """
    Create a stacked bar plot showing fire frequency with bars split by year of last fire.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate year of last fire (assuming current year is around 2021)
    current_year = 2021
    df_plot = df.copy()
    df_plot['year_of_last_fire'] = current_year - df_plot['fire_age_last']
    
    # Create pivot table for stacking
    pivot_table = df_plot.groupby(['num_fires', 'year_of_last_fire']).size().unstack(fill_value=0)
    
    # Create smooth blue color palette from your specified colors
    base_colors = ['#d3d3d3', '#86ceeb', '#1d90ff', '#5941a9']
    
    # Convert hex to RGB for interpolation
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    # Create smooth interpolation between colors
    unique_years = sorted(df_plot['year_of_last_fire'].unique())
    n_colors = len(unique_years)
    
    if n_colors <= len(base_colors):
        # Use subset of base colors
        color_indices = np.linspace(0, len(base_colors)-1, n_colors)
        colors = [base_colors[int(i)] for i in color_indices]
    else:
        # Interpolate smoothly between base colors
        base_rgb = [hex_to_rgb(color) for color in base_colors]
        
        colors = []
        for i in range(n_colors):
            # Map position to color segments
            position = i / (n_colors - 1) * (len(base_colors) - 1)
            segment = int(position)
            fraction = position - segment
            
            if segment >= len(base_colors) - 1:
                colors.append(base_colors[-1])
            else:
                # Interpolate between two adjacent colors
                rgb1 = base_rgb[segment]
                rgb2 = base_rgb[segment + 1]
                interpolated_rgb = tuple(
                    rgb1[j] * (1 - fraction) + rgb2[j] * fraction 
                    for j in range(3)
                )
                colors.append(rgb_to_hex(interpolated_rgb))
    
    # Create color mapping for years
    year_colors = {year: colors[i] for i, year in enumerate(unique_years)}
    plot_colors = [year_colors[year] for year in pivot_table.columns]
    
    # Create stacked bar plot - FIX: Use proper x-positions
    x_positions = range(len(pivot_table.index))  # This creates [0, 1, 2, ...]
    
    # Create first layer
    bars = ax.bar(x_positions, pivot_table.iloc[:, 0], 
                  width=0.6, color=plot_colors[0], edgecolor='black', 
                  linewidth=0.5, zorder=3)
    
    # Add labels to first segment
    for bar_idx, value in enumerate(pivot_table.iloc[:, 0]):
        if value > 0:
            year_label = str(int(pivot_table.columns[0]))
            ax.text(bar_idx, value/2, year_label, 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Add remaining stacked segments
    bottom_values = pivot_table.iloc[:, 0].values.copy()
    
    for col_idx in range(1, len(pivot_table.columns)):
        bars = ax.bar(x_positions, pivot_table.iloc[:, col_idx], 
                     bottom=bottom_values, width=0.6, 
                     color=plot_colors[col_idx], edgecolor='black', 
                     linewidth=0.5, zorder=3)
        
        # Add labels to segments if they're large enough to be visible
        for bar_idx, (bar, value) in enumerate(zip(bars, pivot_table.iloc[:, col_idx])):
            if value > 0:  # Only label non-zero segments
                year_label = str(int(pivot_table.columns[col_idx]))
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_y() + bar.get_height()/2,
                       year_label, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
        
        bottom_values += pivot_table.iloc[:, col_idx].values
    
    # Formatting
    ax.set_xlabel('Fire Frequency', fontsize=12)
    ax.set_ylabel('Number of AOIs', fontsize=12)
    # ax.set_title('Distribution of AOIs by Fire Frequency and Year of Last Fire', 
                # fontsize=14, fontweight='bold')
    
    # Set x-axis labels - FIX: Use fire frequency values as labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(freq)) for freq in pivot_table.index])
    # Remove y-axis tick marks (but keep labels)
    ax.tick_params(axis='y', length=0)
    
    # Grid in background (horizontal only)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Remove vertical frame parts (spines)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add total counts on top of bars
    for i, freq in enumerate(pivot_table.index):
        total = pivot_table.loc[freq].sum()
        ax.text(i, total + total * 0.01, f'{total}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print breakdown
    unique_years = sorted(df_plot['year_of_last_fire'].unique())
    print(f"\nTotal AOIs: {pivot_table.sum().sum()}")
    print(f"Year range: {unique_years[0]} - {unique_years[-1]}")
    print(f"Number of different fire years: {len(unique_years)}")
    
    freq_totals = pivot_table.sum(axis=1)
    print("\nAOIs by fire frequency:")
    for freq, total in freq_totals.items():
        print(f"  {freq} fires: {total} AOIs")


# RUN ALL ANALYSES
plot_edges_by_fire_frequency(df_clean)
plot_edges_vs_recency(df_clean)

plot_stacked_fire_frequency_by_recency(df_clean)


model_main = linear_model_main_effects(df_clean)
model_inter = linear_model_with_interaction(df_clean)
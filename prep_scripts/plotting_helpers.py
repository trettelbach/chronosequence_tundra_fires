import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def add_params_graph(G, edge_param_dict):
    ''' take entire transect dictionary and
    the original graph G and add the mean/median
    parameter values to the graph edges.

    :param G: trough network graph created
    from skeleton
    :param edge_param_dict: dictionary with
    - key: edge (s, e) and
    - value: list with
        - mean width [m]
        - median width [m]
        - mean depth [m]
        - median depth [m]
        - mean r2
        - median r2
        - ratio of considered transects/trough
        - ratio of water-filled troughs
    :return : graph with added edge_param_dict
    parameters added as edge weights.
    '''
    num_emp = 0
    num_full = 0

    # iterate through all graph edges
    for (s, e) in G.edges():
        # and retrieve information on the corresponding edges from the dictionary
        if (s, e) in edge_param_dict:  # TODO: apparently some (xx) edges aren't in the edge_param_dict. check out why
            G[s][e]['mean_width'] = edge_param_dict[(s, e)][0]
            G[s][e]['median_width'] = edge_param_dict[(s, e)][1]
            G[s][e]['mean_depth'] = edge_param_dict[(s, e)][2]
            G[s][e]['median_depth'] = edge_param_dict[(s, e)][3]
            G[s][e]['mean_r2'] = edge_param_dict[(s, e)][4]
            G[s][e]['median_r2'] = edge_param_dict[(s, e)][5]
            G[s][e]['considered_trans'] = edge_param_dict[(s, e)][6]
            G[s][e]['water_filled'] = edge_param_dict[(s, e)][7]
            num_full += 1
        else:
            # print("{} was empty... (border case maybe?)".format(str((s, e))))
            num_emp += 1
    print(num_emp, num_full)


def plot_transect_locations(transect_dict, dem):
    emp = np.zeros(dem.shape)
    print(emp.shape)
    # i = 0
    print(type(transect_dict))
    for outer_keys, transects in transect_dict.items():
        # print(outer_keys)
        # if i == 15 or i == 25 or i == 150:
        for tr_coord, infos in transects.items():
            # print(tr_coord, infos)
            if len(infos[0]) > 0:
                # color the skeleton by fit (r2)
                for i in infos[1]:
                    if i[0] < 300 and i[1] < 300:
                        emp[i[0], i[1]] = 2

                emp[tr_coord[0], tr_coord[1]] = 11

    plt.figure()
    plt.imshow(emp)
    plt.colorbar()
    plt.clim(vmax=2, vmin=0)
    plt.title("plot the transects ALL")

    plt.savefig('/all_transects.png', dpi=300)


def load_fire_data(csv_path):
    """
    Load fire data from CSV file and prepare for plotting.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
        
    Returns
    -------
    pandas.DataFrame
        Processed dataframe with fire data
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Fill NaN values in fire_age_last with -4
    df['fire_age_last'] = df['fire_age_last'].fillna(-4)
    
    # Create combined region column
    df['region_combined'] = df['region'].replace({'csp': 'csp+bkl', 'bkl': 'csp+bkl'})
    
    print(f"Loaded {len(df)} records")
    print(f"Regions: {df['region_combined'].value_counts()}")
    print(f"Fire age last range: {df['fire_age_last'].min()} to {df['fire_age_last'].max()}")
    print(f"Number of edges range: {df['number_of_edges'].min()} to {df['number_of_edges'].max()}")
    
    return df


def fit_quadratic_regression_with_uncertainty(X, y, confidence_level=0.95):
    """
    Fit quadratic regression and calculate prediction intervals.
    
    Parameters
    ----------
    X : array-like
        Independent variable (fire age)
    y : array-like  
        Dependent variable (number of edges)
    confidence_level : float
        Confidence level for uncertainty bands (default: 0.95)
        
    Returns
    -------
    model : sklearn Pipeline
        Fitted quadratic model
    r2 : float
        R² score
    prediction_func : function
        Function to calculate predictions and uncertainty
    """
    
    # Create quadratic polynomial features
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # Fit the model
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate R²
    r2 = r2_score(y, y_pred)
    
    # Calculate residual standard error for uncertainty estimation
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    residual_std = np.sqrt(mse)
    
    # Calculate degrees of freedom
    n = len(y)
    p = 3  # Number of parameters in quadratic model (intercept + x + x²)
    dof = n - p
    
    # t-value for confidence interval
    from scipy import stats
    alpha = 1 - confidence_level
    t_val = stats.t.ppf(1 - alpha/2, dof)
    
    def prediction_with_uncertainty(X_new):
        """Calculate predictions with uncertainty bands."""
        y_pred_new = model.predict(X_new)
        
        # For simplicity, use constant uncertainty based on residual std
        # In practice, this could be more sophisticated (considering leverage, etc.)
        uncertainty = t_val * residual_std
        
        lower_bound = y_pred_new - uncertainty
        upper_bound = y_pred_new + uncertainty
        
        return y_pred_new, lower_bound, upper_bound
    
    return model, r2, residual_std, prediction_with_uncertainty


def plot_edges_vs_fire_age_expanded_grid(df, figsize=(12, 16)):
    """
    Create expanded 4x2 grid of plots for detailed regional analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with fire data
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure, numpy.ndarray
    """
    
    # Prepare data
    regions = df['region_combined'].unique()
    colors = ['#69b437', '#165a54']  # Green for SP, Dark Green for NRV

    # Define variables for each row with y-axis limits
    row_variables = {
        0: {'var': 'number_of_edges', 'label': 'Number of troughs', 'transform': lambda x: x, 'ylim': (0, 3500)},
        1: {'var': 'total_channel_length_m', 'label': 'Total network channel length (km)', 'transform': lambda x: x / 1000, 'ylim': (0, 35)},
        2: {'var': 'num_connected_comp', 'label': 'No. of isolated networks', 'transform': lambda x: x, 'ylim': (0, 125)},
        3: {'var': 'perc_water_fill', 'label': 'Percentage of water-filled troughs', 'transform': lambda x: x*100, 'ylim': (0, 60)}
    }
    
    # Map regions to proper names and sample counts
    region_info = {}
    for region in regions:
        region_data = df[df['region_combined'] == region]
        if 'csp' in region or 'bkl' in region:
            region_info[region] = {'name': 'Seward Peninsula', 'count': len(region_data), 'color': colors[0]}
        else:  # noa
            region_info[region] = {'name': 'Noatak River Valley', 'count': len(region_data), 'color': colors[1]}
    
    # Create figure with 4 rows, 2 columns
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    
    # Set common x-axis limits
    x_min, x_max = -6, 69
    
    # Define subplot labels - by column: first column a-d, second column e-h
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    for row in range(4):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            region_data = df[df['region_combined'] == region]
            info = region_info[region]
            
            # Calculate subplot label index: first column (col=0) gets indices 0-3, second column (col=1) gets indices 4-7
            label_index = col * 4 + row
            subplot_label = subplot_labels[label_index]
            
            # Get current row's variable info
            var_info = row_variables[row]
            var_name = var_info['var']
            y_label = var_info['label']
            transform_func = var_info['transform']
            ylim = var_info['ylim']
            
            # Transform the data
            region_data = region_data.copy()
            region_data[f'{var_name}_transformed'] = transform_func(region_data[var_name])
            
            # BOXPLOTS
            all_fire_ages = sorted(region_data['fire_age_last'].unique())
            box_width = 1.8
            
            boxplot_data = []
            boxplot_positions = []
            
            for fire_age in all_fire_ages:
                age_data = region_data[region_data['fire_age_last'] == fire_age][f'{var_name}_transformed']
                if len(age_data) > 0:
                    boxplot_data.append(age_data.values)
                    boxplot_positions.append(fire_age)
            
            # FIT QUADRATIC REGRESSION FIRST (so it appears behind boxplots)
            regression_data = region_data[region_data['fire_age_last'] != -4].copy()
            
            if len(regression_data) > 5:  # Need enough points for quadratic regression
                X = regression_data['fire_age_last'].values.reshape(-1, 1)
                y = regression_data[f'{var_name}_transformed'].values
                
                # Fit quadratic regression with uncertainty
                model, r2, residual_std, pred_func = fit_quadratic_regression_with_uncertainty(X, y, confidence_level=0.80)
                
                # Create smooth curve for plotting (extend across full x range)
                x_smooth = np.linspace(max(X.min(), x_min), min(X.max(), x_max), 100).reshape(-1, 1)
                y_pred, y_lower, y_upper = pred_func(x_smooth)
                
                # Plot uncertainty bands first (behind everything)
                ax.fill_between(x_smooth.flatten(), y_lower, y_upper, 
                               color='gray', alpha=0.1, zorder=1)
                
                # Plot regression line
                ax.plot(x_smooth, y_pred, '-', color='gray', linewidth=2, alpha=0.6, zorder=2)
                
            elif len(regression_data) > 2:
                # Fall back to linear if not enough points for quadratic
                X = regression_data['fire_age_last'].values.reshape(-1, 1)
                y = regression_data[f'{var_name}_transformed'].values
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                x_smooth = np.linspace(max(X.min(), x_min), min(X.max(), x_max), 100).reshape(-1, 1)
                y_smooth = model.predict(x_smooth)
                
                ax.plot(x_smooth, y_smooth, '--', color='gray', linewidth=2, alpha=0.8, zorder=2)
            else:
                r2 = None  # No regression possible
            
            # BOXPLOTS (on top of regression)
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, positions=boxplot_positions, 
                            patch_artist=True, widths=box_width, showfliers=False, zorder=3)
                
                # Color the boxplots
                for patch in bp['boxes']:
                    patch.set_facecolor(info['color'])
                    patch.set_alpha(0.7)
                
                # Make median lines black
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(1.5)
            
            # ADD SCATTER POINTS ON TOP OF BOXPLOTS
            for fire_age in all_fire_ages:
                age_data = region_data[region_data['fire_age_last'] == fire_age]
                if len(age_data) > 0:
                    # Add small random jitter to x-position
                    jitter = np.random.normal(0, box_width/6, len(age_data))
                    x_positions = [fire_age] * len(age_data) + jitter
                    
                    # Plot scatter points
                    ax.scatter(x_positions, age_data[f'{var_name}_transformed'], 
                              color='black', alpha=0.4, s=20, zorder=4)
            
            # ADD SUBPLOT LABEL IN TOP LEFT CORNER
            ax.text(0.02, 0.98, f'({subplot_label})', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0))
            
            # Set titles only for top row
            if row == 0:
                ax.set_title(f'{info["name"]} (n={info["count"]})', fontsize=12, fontweight='bold')
            
            # Set x-axis labels only for bottom row
            if row == 3:
                ax.set_xlabel('Years since last fire', fontsize=11)
            
            # Set y-axis labels only for leftmost column
            if col == 0:
                ax.set_ylabel(y_label, fontsize=11)
            
            # Customize plot
            ax.grid(True, alpha=0.3)
            
            # Create custom labels
            x_labels = []
            for age in all_fire_ages:
                if age == -4:
                    x_labels.append('unburned')
                else:
                    x_labels.append(f'{int(age)}')

            ax.set_xticklabels(x_labels, fontsize=9)

            # Set x-axis limits and ticks
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(all_fire_ages)

            # Add vertical line at x=0 with "fire" label
            ax.axvline(x=0, color='#FF5C00', linestyle='--', alpha=1, zorder=1)

            # Add vertical "fire" label at x=0
            y_mid = ylim[0] + (ylim[1] - ylim[0]) * 0.1  # Position at 10% of y-axis height
            ax.text(1, y_mid, 'fire', rotation=90, verticalalignment='center', 
                    horizontalalignment='right', 
                    bbox=dict(edgecolor="none", facecolor='white', alpha=0.8),
                    color='black', fontsize=9, alpha=1)
            
            # Set y-axis limits
            ax.set_ylim(ylim)
            
            # Add R² text to each subplot (moved to bottom right)
            if r2 is not None:
                ax.text(0.98, 0.05, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(edgecolor="none", facecolor='white', alpha=0.8),
                       fontsize=10)
    
    plt.tight_layout()
    return fig, axes


def process_specific_transect_files(folder_path):
    """
    Process transect files for specific XXX values from the arf_aoi_XXX_skel.tif list.
    -- the ones that were successfully sampled in HCP terrain
    
    Parameters
    ----------
    folder_path : str
        Path to the dicts_avg folder
        
    Returns
    -------
    dict
        Dictionary with XXX as keys and median values as values
    """
    
    # List of XXX values from arf_aoi_XXX_skel.tif files -- the ones that were sampled in HCP terrain
    xxx_list = [
        "001", "003", "004", "006", "007", "009", "010", "011", "013", "016", 
        "017", "019", "020", "021", "024", "026", "027", "028", "029", "031", 
        "033", "034", "035", "036", "037", "040", "041", "042", "043", "044", 
        "045", "046", "047", "048", "052", "054", "055", "057", "059", "061", 
        "067", "068", "069", "071", "074", "075", "076", "078", "083", "085", 
        "092", "093", "100", "104", "106", "110", "113", "128", "129", "137", 
        "141", "142", "143", "146", "148", "149", "150", "151", "152", "158", 
        "162", "165", "166", "170", "172", "181", "197", "205", "211", "212", 
        "216", "217", "221", "224", "227", "231", "237", "246", "247", "252", 
        "254", "261", "263", "265", "266", "272", "275", "278", "280", "288", 
        "295", "306", "310", "313", "315", "328"
    ]
    
    results = {}
    
    for xxx in xxx_list:
        pkl_file = os.path.join(folder_path, f"transect_dict_avg_{xxx}.pkl")
        
        if not os.path.exists(pkl_file):
            print(f"File not found: transect_dict_avg_{xxx}.pkl")
            continue
            
        try:
            # Read the pickle file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict):
                print(f"Warning: transect_dict_avg_{xxx}.pkl - Expected dictionary, got {type(data)}")
                continue
            
            values_3 = []
            
            for key, value in data.items():
                try:
                    # Check if value has at least 4 elements (index 0, 1, 2, 3)
                    if isinstance(value, (list, tuple)) and len(value) > 2:
                        # Extract value[2] and convert to float if possible
                        val_3 = value[7]
                        if isinstance(val_3, (int, float)):
                            values_3.append(float(val_3))
                        
                except Exception:
                    continue
            
            if values_3:
                median_val = np.nanmedian(values_3)
                results[xxx] = median_val
                print(f"Median of value[3] for XXX={xxx}: {median_val:.6f}")
            else:
                print(f"Warning: No valid numeric value[3] entries found in transect_dict_avg_{xxx}.pkl")
                
        except Exception as e:
            print(f"Error processing transect_dict_avg_{xxx}.pkl: {e}")
            continue
    
    return results


if __name__ == "__main__":
    
    # File paths
    csv_path = "/aois_and_fires_good.csv"
    dicts_avg_folder = "results/dicts_avg"  # Folder containing average dicts (can be found on Zenodo)

    df = load_fire_data(csv_path)

    # Then continue with your plotting
    print("\nCreating expanded 4x2 grid plots...")
    fig2, axes2 = plot_edges_vs_fire_age_expanded_grid(df)
    plt.show()

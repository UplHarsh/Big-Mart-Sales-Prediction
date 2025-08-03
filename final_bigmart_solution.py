import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    data_path = "data"
    
    # Load data
    train_data = pd.read_csv(f'{data_path}/train.csv')
    test_data = pd.read_csv(f'{data_path}/test.csv')
    
    print(f"   - Training data: {train_data.shape}")
    print(f"   - Test data: {test_data.shape}")
    
    # Dataset overview
    print(f"\nDataset Overview:")
    print(f"   - Training samples: {len(train_data):,}")
    print(f"   - Test samples: {len(test_data):,}")
    print(f"   - Features: {len(train_data.columns) - 1}")
    print(f"   - Unique items: {train_data['Item_Identifier'].nunique():,}")
    print(f"   - Unique outlets: {train_data['Outlet_Identifier'].nunique()}")
    
    # Missing values analysis
    print(f"\nMissing Values Analysis:")
    
    train_missing = train_data.isnull().sum()
    test_missing = test_data.isnull().sum()
    
    if train_missing.sum() > 0:
        print("   - Training data:")
        for col, count in train_missing[train_missing > 0].items():
            pct = (count / len(train_data)) * 100
            print(f"     - {col}: {count:,} ({pct:.1f}%)")
    
    if test_missing.sum() > 0:
        print("   - Test data:")
        for col, count in test_missing[test_missing > 0].items():
            pct = (count / len(test_data)) * 100
            print(f"     - {col}: {count:,} ({pct:.1f}%)")
    
    # Target variable analysis
    print(f"\nTarget Variable Analysis (Item_Outlet_Sales):")
    
    sales = train_data['Item_Outlet_Sales']
    stats = sales.describe()
    
    print(f"   - Statistics:")
    print(f"     - Mean: {stats['mean']:,.2f}")
    print(f"     - Median: {stats['50%']:,.2f}")
    print(f"     - Std Dev: {stats['std']:,.2f}")
    print(f"     - Range: {stats['min']:,.2f} - {stats['max']:,.2f}")
    print(f"     - Skewness: {sales.skew():.2f}")
    print(f"     - Kurtosis: {sales.kurtosis():.2f}")
    
    # Key feature analysis
    print(f"\nKey Feature Analysis:")
    
    # Outlet analysis
    outlet_stats = train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].agg(['mean', 'count', 'std'])
    print(f"   - Outlet Types Performance:")
    for outlet_type, stats in outlet_stats.iterrows():
        print(f"     - {outlet_type}: {stats['mean']:,.0f} avg (n={stats['count']:,})")
    
    # Item type analysis
    item_top = train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False).head(5)
    print(f"   - Top 5 Item Types by Average Sales:")
    for item_type, avg_sales in item_top.items():
        print(f"     - {item_type}: {avg_sales:,.0f}")
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Sales distribution
    ax1 = fig.add_subplot(gs[0, 0])
    train_data['Item_Outlet_Sales'].hist(bins=50, alpha=0.8, color='skyblue', ax=ax1)
    ax1.axvline(train_data['Item_Outlet_Sales'].mean(), color='red', linestyle='--', label='Mean')
    ax1.axvline(train_data['Item_Outlet_Sales'].median(), color='green', linestyle='--', label='Median')
    ax1.set_title('Sales Distribution')
    ax1.set_xlabel('Sales')
    ax1.legend()
    
    # 2. Sales by Outlet Type
    ax2 = fig.add_subplot(gs[0, 1])
    outlet_sales = train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
    bars = ax2.bar(range(len(outlet_sales)), outlet_sales.values, color='lightcoral')
    ax2.set_title('Avg Sales by Outlet Type')
    ax2.set_xticks(range(len(outlet_sales)))
    ax2.set_xticklabels([t.replace('Supermarket Type', 'SM') for t in outlet_sales.index], rotation=45)
    
    # 3. Sales vs MRP scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(train_data['Item_MRP'], train_data['Item_Outlet_Sales'], 
               alpha=0.5, s=10, color='green')
    ax3.set_title('Sales vs MRP')
    ax3.set_xlabel('MRP')
    ax3.set_ylabel('Sales')
    
    # 4. Top Item Types
    ax4 = fig.add_subplot(gs[0, 3])
    top_items = train_data.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=True).tail(8)
    ax4.barh(range(len(top_items)), top_items.values, color='purple')
    ax4.set_title('Top Item Types by Avg Sales')
    ax4.set_yticks(range(len(top_items)))
    ax4.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_items.index])
    
    # 5. Sales by Outlet Size
    ax5 = fig.add_subplot(gs[1, 0])
    size_sales = train_data.groupby('Outlet_Size')['Item_Outlet_Sales'].mean()
    ax5.bar(size_sales.index, size_sales.values, color='orange')
    ax5.set_title('Sales by Outlet Size')
    
    # 6. Fat Content Distribution
    ax6 = fig.add_subplot(gs[1, 1])
    fat_counts = train_data['Item_Fat_Content'].value_counts()
    ax6.pie(fat_counts.values, labels=fat_counts.index, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Fat Content Distribution')
    
    # 7. Establishment Year Trend
    ax7 = fig.add_subplot(gs[1, 2])
    year_trend = train_data.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean()
    ax7.plot(year_trend.index, year_trend.values, marker='o', color='red')
    ax7.set_title('Sales by Establishment Year')
    ax7.set_xlabel('Year')
    ax7.grid(True, alpha=0.3)
    
    # 8. Item Visibility
    ax8 = fig.add_subplot(gs[1, 3])
    vis_data = train_data[train_data['Item_Visibility'] > 0]['Item_Visibility']
    ax8.hist(vis_data, bins=30, alpha=0.7, color='brown')
    ax8.set_title('Item Visibility (>0)')
    ax8.set_xlabel('Visibility')
    
    # 9. Correlation Heatmap
    ax9 = fig.add_subplot(gs[2, :2])
    numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']
    corr_matrix = train_data[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax9, cbar_kws={'shrink': 0.8})
    ax9.set_title('Feature Correlation Matrix')
    
    # 10. Sales by Location Type
    ax10 = fig.add_subplot(gs[2, 2])
    location_sales = train_data.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean()
    ax10.bar(location_sales.index, location_sales.values, color='lightgreen')
    ax10.set_title('Sales by Location Type')
    
    # 11. Weight vs Sales
    ax11 = fig.add_subplot(gs[2, 3])
    weight_data = train_data.dropna(subset=['Item_Weight'])
    ax11.scatter(weight_data['Item_Weight'], weight_data['Item_Outlet_Sales'], alpha=0.5, s=10, color='blue')
    ax11.set_title('Weight vs Sales')
    ax11.set_xlabel('Weight')
    
    # 12. Sales Distribution by Key Categories
    ax12 = fig.add_subplot(gs[3, :])
    # Box plot of sales by outlet type
    outlet_types = train_data['Outlet_Type'].unique()
    sales_data = [train_data[train_data['Outlet_Type'] == ot]['Item_Outlet_Sales'] for ot in outlet_types]
    bp = ax12.boxplot(sales_data, labels=[ot.replace('Supermarket Type', 'SM') for ot in outlet_types], patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax12.set_title('Sales Distribution by Outlet Type')
    ax12.set_ylabel('Sales')
    ax12.tick_params(axis='x', rotation=45)
    
    plt.suptitle('BigMart Sales - Comprehensive Exploratory Data Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig('comprehensive_eda.png', dpi=300, bbox_inches='tight')
    
    # Preprocessing
    # Combine for consistent preprocessing
    train_copy = train_data.copy()
    test_copy = test_data.copy()
    
    train_copy['source'] = 'train'
    test_copy['source'] = 'test'
    test_copy['Item_Outlet_Sales'] = 0
    
    combined = pd.concat([train_copy, test_copy], ignore_index=True)
    print(f"Combined dataset: {combined.shape}")
    
    # Handle missing values
    print("\nHandling missing values intelligently...")
    
    # Item_Weight: Multiple strategies
    print("   - Item_Weight missing values...")
    
    # Strategy 1: By Item_Identifier
    weight_by_item = combined.groupby('Item_Identifier')['Item_Weight'].mean()
    combined['Item_Weight'] = combined.apply(
        lambda x: weight_by_item.get(x['Item_Identifier'], x['Item_Weight']) if pd.isna(x['Item_Weight']) else x['Item_Weight'], 
        axis=1
    )
    
    # Strategy 2: By Item_Type
    weight_by_type = combined.groupby('Item_Type')['Item_Weight'].mean()
    combined['Item_Weight'] = combined.apply(
        lambda x: weight_by_type[x['Item_Type']] if pd.isna(x['Item_Weight']) else x['Item_Weight'], 
        axis=1
    )
    
    # Strategy 3: Overall mean
    combined['Item_Weight'].fillna(combined['Item_Weight'].mean(), inplace=True)
    
    # Outlet_Size: Context-aware imputation
    print("   - Outlet_Size missing values...")
    
    # Use mode by outlet type and location
    for outlet_type in combined['Outlet_Type'].unique():
        for location in combined['Outlet_Location_Type'].unique():
            mask = (combined['Outlet_Type'] == outlet_type) & (combined['Outlet_Location_Type'] == location)
            subset = combined[mask]
            
            if len(subset) > 0 and not subset['Outlet_Size'].mode().empty:
                mode_size = subset['Outlet_Size'].mode()[0]
                null_mask = mask & combined['Outlet_Size'].isna()
                combined.loc[null_mask, 'Outlet_Size'] = mode_size
    
    # Fill remaining nulls with mode by outlet type
    for outlet_type in combined['Outlet_Type'].unique():
        type_mask = combined['Outlet_Type'] == outlet_type
        if not combined[type_mask]['Outlet_Size'].mode().empty:
            mode_size = combined[type_mask]['Outlet_Size'].mode()[0]
            null_mask = type_mask & combined['Outlet_Size'].isna()
            combined.loc[null_mask, 'Outlet_Size'] = mode_size
    
    # Clean and standardize
    # Standardize Item_Fat_Content
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular',
        'LOW FAT': 'Low Fat', 'REGULAR': 'Regular'
    })
    
    # Engineer features
    # Time-based features
    combined['Outlet_Years'] = 2013 - combined['Outlet_Establishment_Year']
    combined['Outlet_Age_Category'] = pd.cut(combined['Outlet_Years'], 
                                       bins=[0, 8, 15, 30], 
                                       labels=['New', 'Established', 'Mature'])
    
    combined['Item_Category'] = combined['Item_Identifier'].str[:2]
    combined['Item_Number'] = pd.to_numeric(combined['Item_Identifier'].str[2:], errors='coerce').fillna(0).astype(int)
    
    # Price and weight features
    combined['Price_per_Weight'] = combined['Item_MRP'] / combined['Item_Weight']
    combined['Has_Visibility'] = (combined['Item_Visibility'] > 0).astype(int)
    
    visibility_means = combined[combined['Item_Visibility'] > 0].groupby('Item_Type')['Item_Visibility'].mean()
    combined['Item_Visibility_Adjusted'] = combined.apply(
        lambda x: visibility_means[x['Item_Type']] if x['Item_Visibility'] == 0 else x['Item_Visibility'], 
        axis=1
    )
    
    combined['MRP_Category'] = pd.qcut(combined['Item_MRP'], q=5, 
                                 labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
    combined['Weight_Category'] = pd.qcut(combined['Item_Weight'], q=4, 
                                    labels=['Light', 'Medium', 'Heavy', 'VeryHeavy'])
    
    # Create advanced features
    train_mask = combined['source'] == 'train'
    if train_mask.sum() > 0:
        store_stats = combined[train_mask].groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg([
            'mean', 'std', 'count'
        ])
        store_stats.columns = ['Outlet_Avg_Sales', 'Outlet_Sales_Std', 'Outlet_Item_Count']
        
        combined = combined.merge(store_stats, left_on='Outlet_Identifier', right_index=True, how='left')
        
        # Fill test data
        overall_mean = combined[train_mask]['Item_Outlet_Sales'].mean()
        overall_std = combined[train_mask]['Item_Outlet_Sales'].std()
        overall_count = combined[train_mask].groupby('Outlet_Identifier').size().mean()
        
        combined['Outlet_Avg_Sales'].fillna(overall_mean, inplace=True)
        combined['Outlet_Sales_Std'].fillna(overall_std, inplace=True)
        combined['Outlet_Item_Count'].fillna(overall_count, inplace=True)
    
    # Market positioning
    combined['Market_Position'] = 'Standard'
    
    high_mrp_mask = combined['Item_MRP'] > combined['Item_MRP'].quantile(0.8)
    low_fat_mask = combined['Item_Fat_Content'] == 'Low Fat'
    combined.loc[high_mrp_mask & low_fat_mask, 'Market_Position'] = 'Premium'
    
    low_mrp_mask = combined['Item_MRP'] < combined['Item_MRP'].quantile(0.2)
    combined.loc[low_mrp_mask, 'Market_Position'] = 'Budget'
    
    # Interaction features
    combined['Outlet_Item_Combo'] = combined['Outlet_Type'] + '_' + combined['Item_Type']
    combined['Size_Location_Combo'] = combined['Outlet_Size'].fillna('Unknown') + '_' + combined['Outlet_Location_Type']
    
    # Encode categorical variables
    print("\nEncoding categorical features...")
    
    categorical_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 
        'Outlet_Type', 'Item_Category', 'Outlet_Age_Category', 'MRP_Category', 
        'Weight_Category', 'Market_Position', 'Outlet_Item_Combo', 'Size_Location_Combo'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        if col in combined.columns:
            le = LabelEncoder()
            # Convert to string first, then handle nulls
            combined[col] = combined[col].astype(str)
            combined[col] = combined[col].replace('nan', 'Unknown')
            combined[col] = le.fit_transform(combined[col])
            label_encoders[col] = le
            print(f"   - {col}: {len(le.classes_)} categories")
    
    processed_data = combined
    print("Advanced preprocessing completed!")
    
    # Initialize models
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.005,
            random_state=42,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=300, learning_rate=0.01, random_state=42),
        'LinearRegression': LinearRegression(),
        'MLPRegressor': MLPRegressor(
            hidden_layer_sizes=(275, 108,100), max_iter=2500, random_state=42,
            learning_rate_init=0.0002, early_stopping=True
        ),
        'Ridge': Ridge(alpha=5.0),
        'HuberRegressor': HuberRegressor(epsilon=1.35, max_iter=200),
        'TheilSenRegressor': TheilSenRegressor(random_state=42, max_subpopulation=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'KNeighbors': KNeighborsRegressor(n_neighbors=8, weights='distance')
    }
    
    print(f"Initialized {len(models)} models:")
    for i, name in enumerate(models.keys(), 1):
        print(f"   {i:2d}. {name}")
    
    # Prepare features
    print(f"\nPreparing features...")
    
    # Core feature set
    feature_cols = [
        # Basic features
        'Item_Weight', 'Item_Fat_Content', 'Item_Visibility_Adjusted', 'Item_Type', 
        'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 
        'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Years',
        
        # Engineered features
        'Item_Category', 'Item_Number', 'Price_per_Weight', 'Has_Visibility',
        'MRP_Category', 'Weight_Category', 'Market_Position', 'Outlet_Age_Category',
        
        # Advanced features
        'Outlet_Avg_Sales', 'Outlet_Sales_Std', 'Outlet_Item_Count',
        'Outlet_Item_Combo', 'Size_Location_Combo'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in processed_data.columns]
    
    # Split data
    train_data_processed = processed_data[processed_data['source'] == 'train']
    test_data_processed = processed_data[processed_data['source'] == 'test']
    
    X_train = train_data_processed[available_features]
    y_train = train_data_processed['Item_Outlet_Sales']
    X_test = test_data_processed[available_features]
    
    feature_names = available_features
    
    print(f"   - Features: {len(available_features)}")
    print(f"   - Training: {X_train.shape}")
    print(f"   - Test: {X_test.shape}")
    
    # Train and evaluate models
    print(f"\nTraining and evaluating models...")
    
    # Validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train
            model.fit(X_train_split, y_train_split)
            
            # Predict
            train_pred = model.predict(X_train_split)
            val_pred = model.predict(X_val_split)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_split, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
            val_r2 = r2_score(y_val_split, val_pred)
            val_mae = mean_absolute_error(y_val_split, val_pred)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae
            }
            
            print(f"   - Train RMSE: {train_rmse:8.2f}")
            print(f"   - Val RMSE:   {val_rmse:8.2f}")
            print(f"   - Val R²:     {val_r2:8.4f}")
            print(f"   - Val MAE:    {val_mae:8.2f}")
            
        except Exception as e:
            print(f"   - Error: {str(e)}")
            continue
    
    if results:
        # Select best model
        best_name = min(results.keys(), key=lambda x: results[x]['val_rmse'])
        best_model = results[best_name]['model']
        
        print(f"\nBest model: {best_name}")
        print(f"   RMSE: {results[best_name]['val_rmse']:.2f}")
        
        # Retrain on full data
        print(f"Retraining {best_name} on full dataset...")
        best_model.fit(X_train, y_train)
        fitted_models[best_name] = best_model
    
    # Generate predictions
    predictions = best_model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    print(f"Generated {len(predictions):,} predictions")
    print(f"   - Range: {predictions.min():,.2f} - {predictions.max():,.2f}")
    print(f"   - Mean: {predictions.mean():,.2f}")
    
    # Create submission file
    print(f"\nCreating submission...")
    
    submission = pd.DataFrame({
        'Item_Identifier': test_data['Item_Identifier'],
        'Outlet_Identifier': test_data['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    submission.to_csv('final_submission.csv', index=False)
    
    # Plot model comparison
    print(f"\nCreating model performance visualization...")
    
    models_list = list(results.keys())
    metrics = ['val_rmse', 'val_r2', 'val_mae']
    metric_names = ['RMSE', 'R²', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in models_list]
        
        bars = axes[i].bar(models_list, values, color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_title(f'Model Comparison - {name}')
        axes[i].set_ylabel(name)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Highlight best
        if metric == 'val_r2':  # Higher is better
            best_idx = np.argmax(values)
        else:  # Lower is better
            best_idx = np.argmin(values)
            
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        # Add values
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, height,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print(f"   - Saved: model_performance.png")
    
    # Plot feature importance
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nCreating feature importance plot...")
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(14, 8))
        
        # Top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), importances[top_indices], color='lightblue', alpha=0.8)
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        # Add values
        for i, importance in enumerate(importances[top_indices]):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"   - Saved: feature_importance.png")
        
        # Print top features
        print(f"\nTop 10 Features:")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"   {i+1:2d}. {feature_names[idx]:25} | {importances[idx]:.4f}")
    
    # Final summary
    best_name = min(results.keys(), key=lambda x: results[x]['val_rmse'])
    best_results = results[best_name]
    
    print(f"\nBEST MODEL PERFORMANCE:")
    print(f"   - Model: {best_name}")
    print(f"   - RMSE: {best_results['val_rmse']:,.2f}")
    print(f"   - R²: {best_results['val_r2']:.4f}")
    print(f"   - MAE: {best_results['val_mae']:,.2f}")
    
    # Model ranking
    print(f"\nMODEL RANKING:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
    
    for i, (name, scores) in enumerate(sorted_models, 1):
        print(f"   {i}. {name:<20} | RMSE: {scores['val_rmse']:>7,.0f} | R²: {scores['val_r2']:>6.3f}")
    
    print(f"\nFILES GENERATED:")
    print(f"   - comprehensive_eda.png")
    print(f"   - model_performance.png")
    print(f"   - feature_importance.png")
    print(f"   - final_submission.csv")
    
    print(f"\nSystem completed successfully!")
    return submission

if __name__ == "__main__":
    main()
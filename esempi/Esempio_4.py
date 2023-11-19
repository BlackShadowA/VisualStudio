


def train_pipeline_rf(X,y):
    # X are factors
    # y is output
    # impute missing X by median
    X_prepared = pre_pipeline.fit_transform(X)
    # set cross-validation
    tscv = TimeSeriesSplit(n_splits=10)
    data_split = tscv.split(X_prepared)
    # hyper-parameter space
    param_grid_RF = {
        'n_estimators' : [10,20,50,100,200,500,1000],
        'max_features' : [0.6,0.8,"auto","sqrt"]
    }
    # build random forest model
    rf_model = RandomForestRegressor(random_state=42,n_jobs=-1)
    # gridsearch for the best hyper-parameter
    gs_rf = GridSearchCV(rf_model, param_grid=param_grid_RF, cv=data_split, scoring='neg_mean_squared_error', n_jobs=-1)
    # fit dataset
    gs_rf.fit(X_prepared, y)
    return gs_rf


pre_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])


# hyper-parameter space
    param_grid_RF = {
        'n_estimators' : [10,20,50,100,200,500,1000]
    }
# build random forest model
    rf_model = RandomForestRegressor(random_state=42,n_jobs=-1,max_features=0.6,max_depth=5)
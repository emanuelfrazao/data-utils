from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 



################################## BASIC OPERATIONS ###########################################

#### drop features

class FeatureDroper( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    # Return self
    def fit( self, X, y = None ):
        return self 
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X.drop(self.feature_names, axis=1)




#### transform features

class FeatureTransformer( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, features_transformers: dict):
        self.features_transformers = features_transformers
        self.features = list(features_transformers.keys())
    
    #Return self nothing else to do here    
    def fit( self, X, y = None, **kwargs ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        X_return = X.copy()
        
        # Transform each feature
        for feature, transformer in self.features_transformers.items():
            
            X_return[feature] = X_return[feature].map(transformer)
        
        return X_return




#### costum discretizer

class CostumDiscretizer( BaseEstimator, TransformerMixin ):
    """
    Example:
        CostumDiscretizer(
            'age',
            [
                {
                    'condition': lambda x: x> 30,
                    'value': 'old'
                },
                {
                    'condition': lambda x: (x > 20) & (x < 30),
                    'value': 'young'
                },
                {
                    'condition': lambda x: x <= 20,
                    'value': 'child'
                }
            ],
            'age_group'
        ).fit_transform(df)
    """
    #Class Constructor 
    def __init__( self, feature, buckets, feature_new_name=None):
        """
        buckets: list[dict]: list of dict for each bucket. 
        Each bucket (dict) must have a key named 'condition' with a function filter as value and a key named 'value' with the respective value as value.
        """
        self.feature = feature
        self.buckets = buckets
        self.feature_new_name = feature_new_name if feature_new_name else feature
    
    #Return self nothing else to do here    
    def fit( self, X, y = None, **kwargs ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        X_return = X.copy()
        
        # Transform each condition into the respective value
        for bucket in self.buckets:
            X_return.loc[X[self.feature].apply(bucket['condition']), self.feature] = bucket['value']
        
        return X_return.rename({self.feature: self.feature_new_name}, axis=1)




#### imputer

class ProperImputer( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, features_to_input, missing_values=np.nan, strategy='mean', fill_value=None):
        self.features_to_input = features_to_input
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        
        # Create instance of imputer
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        
    # Return self nothing else to do here    
    def fit( self, X, y = None ):
        
        # Fit scaler
        self.imputer.fit(X[self.features_to_input])
        
        return self
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        # Get the encoding as an array
        imp_array = self.imputer.transform(X[self.features_to_input])

        # Get the encoding as a dataframe
        imp_df = pd.DataFrame(imp_array, index=X.index, columns=self.features_to_input)

        # Substitute in original dataframe
        X_return = X.copy()
        X_return[self.features_to_input] = imp_df

        return X_return




######################################## CATEGORICALS ################################################


#### one-hot-encoder

class ProperOneHotEncoder( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, features_to_encode ):
        self.features_to_encode = features_to_encode
        
        # Create instance of OneHotEncoder
        self.enc = OneHotEncoder(sparse=False)
    
    # Return self    
    def fit( self, X, y = None ):
        
        self.enc.fit(X[self.features_to_encode])        
        return self
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
    
        # Get the encoding as an array
        enc_array = self.enc.transform(X[self.features_to_encode])
        
        # Get the encoding as a dataframe
        column_names = [feature + '_IS_' + str(category) \
                        for i, feature in enumerate(self.features_to_encode) \
                        for category in self.enc.categories_[i]]

        enc_df = pd.DataFrame(enc_array, index=X.index, columns=column_names)
        
        # Substitute in original dataframe
        return X.drop(self.features_to_encode, axis=1).join(enc_df)



#### dumb variable encoder

class ProperDumbVariableEncoder( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, features_to_encode, classes_to_drop=None ):
        self.features_to_encode = features_to_encode 
        self.classes_to_drop = classes_to_drop
        
        # Create instance of OneHotEncoder
        self.enc = OneHotEncoder(
            drop = classes_to_drop if classes_to_drop else 'first', 
            sparse=False
        )
        
    # Return self    
    def fit( self, X, y = None ):
        
        self.enc.fit(X[self.features_to_encode])
        return self
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        

        # Get the encoding as an array
        enc_array = self.enc.transform(X[self.features_to_encode])
        
        # Get the encoding as a dataframe
        column_names = [feature + '_IS_' + str(category) \
                        for i, feature in enumerate(self.features_to_encode) \
                        for category in ([x for x in self.enc.categories_[i] if x != self.enc.drop[i]] 
                                         if self.classes_to_drop 
                                         else self.enc.categories_[i][1:])
                       ]

        enc_df = pd.DataFrame(enc_array, index=X.index, columns=column_names)
        
        # Substitute in original dataframe
        return X.drop(self.features_to_encode, axis=1).join(enc_df)




#### ordinal encoder

class ProperOrdinalEncoder( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, features_to_encode, scaled=True, scaling_factor=None, increasing_order = True, start_at_zero = True ):
        self.features_to_encode = features_to_encode 
        self.increasing_order = increasing_order
        self.start_at_zero = start_at_zero
        self.scaled = scaled
        self.scaling_factor = scaling_factor
        
        # Create instance of OrdinalEncoder
        self.enc = OrdinalEncoder()
    
    # Return self   
    def fit( self, X, y = None ):
        
        self.enc.fit(X[self.features_to_encode])
        self.max_values = [len(classes)-1 for classes in self.enc.categories_]
        
        return self
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        # Get the encoding as an array
        enc_array = self.enc.transform(X[self.features_to_encode]) + (0 if self.start_at_zero else 1)

        if not self.increasing_order:
            enc_array = self.max_values - enc_array + (0 if self.start_at_zero else 1)

        # Scale the encoding (0 to 1)
        if self.scaled:
            enc_array = enc_array / self.max_values
            
            # Multiply by scaling factor
            if self.scaling_factor and isinstance(self.scaling_factor, (int, float)):
                enc_array = enc_array * self.scaling_factor

        # Get the encoding as a DataFrame
        enc_df = pd.DataFrame(enc_array, index=X.index, columns=self.features_to_encode)

        # Substitute in original dataframe
        X_return = X.copy()
        
        X_return[self.features_to_encode] = enc_df
        
        return X_return




#### collapse ordered categorical

class OrderedCollapser( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, feature, collapsed_name, boundary, is_lower_bound=True):
        self.feature = feature
        self.boundary = boundary
        self.is_lower_bound = is_lower_bound
        self.collapsed_name = collapsed_name
    
    # Return self nothing else to do here    
    def fit( self, X, y = None, **kwargs ):
        return self 
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self.feature not in X.columns:
            return X
        else:
            X_return = X.copy()
            
            if self.is_lower_bound:
                X_return.loc[X_return[self.feature] > self.boundary, self.feature] = self.collapsed_name
            else:
                X_return.loc[X_return[self.feature] < self.boundary, self.feature] = self.collapsed_name
            
            return X_return




################################################### NUMERICALS ########################################################


#### discretize numericals

class ProperOrdinalDiscretizer( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature, n_bins, strategy):
        self.feature = feature
        self.n_bins = n_bins
        self.strategy = strategy
        self.disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    
    #Return self nothing else to do here    
    def fit( self, X, y = None, **kwargs ):
        
        self.disc.fit(X[[self.feature]])
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        # Get the encoding as an array
        disc_array = self.disc.transform(X[[self.feature]])
        disc_array = disc_array.reshape((disc_array.shape[1], disc_array.shape[0]))[0]

        #Get the encoding as a dataframe
        disc_df = pd.Series(disc_array, index=X.index)

        # Substitute in original dataframe
        X_return = X.copy()
        X_return[self.feature] = disc_df

        return X_return





#### scale numericals

class ProperScaler( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, features_to_scale, kind_of_scale='standard' ):
        """
            kind_of_scale: one of {'standard', 'minmax', 'maxabs'} for the corresponding scalers
        """
        self.features_to_scale = features_to_scale 
        self.kind_of_scale = kind_of_scale
        
        # Create instance of scaler
        if kind_of_scale == 'standard':
            self.scaler = StandardScaler()
        elif kind_of_scale == 'minmax':
            self.scaler = MinMaxScaler()
        elif kind_of_scale == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            self.scaler = StandardScaler()
        
    # Return self nothing else to do here    
    def fit( self, X, y = None ):
        
        # Fit scaler
        self.scaler.fit(X[self.features_to_scale])
        
        return self
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        # Get the encoding as an array
        ss_array = self.scaler.transform(X[self.features_to_scale])

        # Get the encoding as a dataframe
        ss_df = pd.DataFrame(ss_array, index=X.index, columns=self.features_to_scale)

        # Substitute in original dataframe
        X_return = X.copy()
        X_return[self.features_to_scale] = ss_df

        return X_return



#### combine numericals

class FeaturesCombinerWithNorm( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names, norm_ord=2 ):
        self.feature_names = feature_names
        self.norm_ord = norm_ord
    
    #Return self nothing else to do here    
    def fit( self, X, y = None, **kwargs ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        X_return = X.copy()
        
        column_name = 'combined ' + ' '.join(self.feature_names)
        
        X_return[column_name] = X_return[self.feature_names].apply(lambda x: np.linalg.norm(x, ord=self.norm_ord), axis=1)
        
        return X_return






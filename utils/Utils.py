import numpy as np

def mostUsedWordsByVeracity(df):
    df=df.groupby(['veracity','token'])['token'] \
        .count() \
        .reset_index(name='count') \
        .groupby('veracity') \
        .apply(lambda x: x.sort_values('count', ascending=False).head(5))
    
    return(df)

def calculateLogOfOddsRatio(df):
    # Pivotaje y despivotaje
    tweets_pivot = df.groupby(["veracity","token"])["token"] \
                    .agg(["count"]).reset_index() \
                    .pivot(index = "token" , columns="veracity", values= "count")

    tweets_pivot = tweets_pivot.fillna(value=0)
    tweets_pivot.columns.name = None

    tweets_unpivot = tweets_pivot.melt(value_name='n', var_name='veracity', ignore_index=False)
    tweets_unpivot = tweets_unpivot.reset_index()

    # Selección de los autores elonmusk y mayoredlee
    tweets_unpivot = tweets_unpivot[tweets_unpivot.veracity.isin(['T', 'F'])]

    # Se añade el total de palabras de cada autor
    tweets_unpivot = tweets_unpivot.merge(
                        df.groupby('veracity')['token'].count().rename('N'),
                        how = 'left',
                        on  = 'veracity'
                    )

    # Cálculo de odds y log of odds de cada palabra
    tweets_logOdds = tweets_unpivot.copy()
    tweets_logOdds['odds'] = (tweets_logOdds.n + 1) / (tweets_logOdds.N + 1)
    tweets_logOdds = tweets_logOdds[['token', 'veracity', 'odds']] \
                        .pivot(index='token', columns='veracity', values='odds')
    tweets_logOdds.columns.name = None

    tweets_logOdds['log_odds'] = np.log(tweets_logOdds['T'] / tweets_logOdds['F'])
    tweets_logOdds['abs_log_odds'] = np.abs(tweets_logOdds.log_odds)

    # Si el logaritmo de odds es mayor que cero, significa que es una palabra con
    # mayor probabilidad de ser T. Esto es así porque el ratio sea ha
    # calculado como T/F.
    tweets_logOdds['veracidad_frecuente'] = np.where(tweets_logOdds.log_odds > 0,
                                              "T",
                                              "F"
                                    )
    return(tweets_logOdds)

def topWordsForNewsType(df):
    top_30 = calculateLogOfOddsRatio(df)[['log_odds', 'abs_log_odds', 'veracidad_frecuente']] \
        .groupby('veracidad_frecuente') \
        .apply(lambda x: x.nlargest(15, columns='abs_log_odds').reset_index()) \
        .reset_index(drop=True) \
        .sort_values('abs_log_odds', ascending=False)
    return(top_30[['veracidad_frecuente', 'token', 'log_odds', 'abs_log_odds']])
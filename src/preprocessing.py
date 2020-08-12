import copy
import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook as tqdm

def wareki_to_seireki(wareki):
    """
    西暦を和暦にする（昭和、平成のみ対応）
    
    Parameter:
    ----
    wareki_list: 和暦のSeries
    
    Return:
    -----
    seireki_list: 西暦のSeries
    
    """
    wareki_list = list(wareki.values)
    
    for i in tqdm(range(len(wareki_list))):
        try:
            if re.match('昭和元年', wareki_list[i]):
                wareki_list[i] = 1926

            elif re.match('昭和',wareki_list[i]):
                wareki_list[i] = 1925 + int(re.sub(r'\D','',wareki_list[i]))

            elif re.match('平成元年', wareki_list[i]):
                wareki_list[i] = 1989

            elif re.match('平成',wareki_list[i]):
                wareki_list[i] = 1988 + int(re.sub(r'\D','',wareki_list[i]))
        except:
            continue
            
    return pd.Series(wareki_list)
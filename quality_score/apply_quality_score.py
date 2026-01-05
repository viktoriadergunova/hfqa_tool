import pandas as pd
import numpy as np
from etl.normalization import normalize_vocabulary_series

def calculate_u_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    u_cfg = qc_schema['u_score']['calculation']
    t = qc_schema['u_score']['thresholds']

    relevance = normalize_vocabulary_series(df[u_cfg['relevance_col']])
    
    ROLE_CHILD = u_cfg.get('role_child', '[yes]')
    ROLE_PARENT = u_cfg.get('role_parent', '[no]')

    val_num = pd.to_numeric(df[u_cfg['value_col']], errors='coerce')
    unc_num = pd.to_numeric(df[u_cfg['uncertainty_col']], errors='coerce')

    # 3. Calculate Child Scores based on COV
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = (unc_num / val_num.abs()) * 100

    conditions = [
        (cov < t['U1']),
        (cov >= t['U1']) & (cov < t['U2']),
        (cov >= t['U2']) & (cov < t['U3']),
        (cov >= t['U3'])
    ]
    choices = ["U1", "U2", "U3", "U4"]
    child_raw_scores = np.select(conditions, choices, default="Ux")
    
    # Attach to a temporary column for grouping
    df_temp = df.copy()
    df_temp['temp_u'] = child_raw_scores

    # 4. Inheritance Map: Poorest Child Result per Site (P3)
    rank_map = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 5}
    rev_map = {v: k for k, v in rank_map.items()}
    
    # Filter for children
    children = df_temp[relevance == ROLE_CHILD].copy()
    
    if not children.empty:
        # MAP FIRST: Convert U1-U4 to 1-4
        children['u_rank'] = children['temp_u'].map(rank_map)
        
        # GROUP SECOND: Find the max rank per site name (P3)
        # Then map back to U-strings
        site_poorest = (
            children.groupby(u_cfg['site_name_col'])['u_rank']
            .max()
            .map(rev_map)
            .to_dict()
        )
    else:
        site_poorest = {}

    # 5. Final Role-Based Assignment
    def finalize(idx):
        row_rel = relevance.loc[idx]
        row_site = df.loc[idx, u_cfg['site_name_col']]
        
        if row_rel == ROLE_PARENT:
            # I am the parent row -> I get the inherited worst-case child score
            return site_poorest.get(row_site, "Ux")
        elif row_rel == ROLE_CHILD:
            # I am a child row -> I keep my own calculated score
            return df_temp.loc[idx, 'temp_u']
        return "Ux"

    return pd.Series([finalize(i) for i in df.index], index=df.index)

def inherit_u_score_to_parent(df_child: pd.DataFrame, qc_schema: dict) -> pd.DataFrame:
    # 1. Map scores to numeric ranks
    rank_map = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 5}
    rev_map = {v: k for k, v in rank_map.items()}

    # 2. Use P3 for grouping (Standard IHFC Site Name column)
    site_col = "P3" 
    
    # 3. Filter for relevance based on your C9 column
    rel_col = "C9"
    relevant_df = df_child[df_child[rel_col] == '[yes]'].copy()

    if relevant_df.empty:
        return pd.DataFrame(columns=[site_col, 'parent_quality_U'])

    # find the poorest rank
    relevant_df['u_numeric'] = relevant_df['quality_U'].map(rank_map)

    parent_scores = relevant_df.groupby(site_col)['u_numeric'].max().reset_index()
    
    # 5. Map back to U-strings
    parent_scores['parent_quality_U'] = parent_scores['u_numeric'].map(rev_map)
    
    return parent_scores[[site_col, 'parent_quality_U']]
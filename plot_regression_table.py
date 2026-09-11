"""
Render publication-quality OLS regression tables as matplotlib figures.
Dataset: combined_clean_232.xlsx  (N=232, wave1=168, wave2=64)

Outputs
-------
regression_table.png / .pdf   — full model (M3), all predictors
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Times New Roman', 'DejaVu Serif', 'Georgia']

df_full = pd.read_excel('regression_results.xlsx', sheet_name='Full_results')
df_fit  = pd.read_excel('regression_results.xlsx', sheet_name='Model_fit')

OUTCOMES = ['EMQ', 'EHB', 'ECO', 'MGT', 'OVERALL']

PRED_CONFIG = [
    ('wave2',                'Demographics',        'Survey wave (ref: Wave 1)',        False),
    ('gender_bin',           None,                  'Gender (ref: Male)',               False),
    ('age_num',              None,                  'Age group (ordinal 1-4)',          False),
    ('edu_num',              None,                  'Education (ordinal 1-4)',          False),
    ('dist_num',             None,                  'Distance to park (ordinal 1-5)',   False),
    ('freq_num',             None,                  'Visit frequency (ordinal 1-5)',    False),
    ('Info_Provision_score', 'Participation',       'Information provision score',      False),
    ('Consultation_score',   None,                  'Consultation score',               False),
    ('PAW',                  'Public Awareness',    'PAW composite',                    False),
]

def _stars(sig):
    return {'***':'***','**':'**','*':'*','t':'†','ns':''}[sig]

def _fmt_beta(b_std, sig):
    if b_std is None or (isinstance(b_std, float) and np.isnan(b_std)): return '-'
    sign   = '-' if b_std < 0 else ' '
    digits = f'{round(abs(b_std)*1000):03d}'
    return f'{sign}.{digits}{_stars(sig)}'

def _fmt_bse(b, se):
    if b is None or (isinstance(b, float) and np.isnan(b)): return '-'
    return f"{'−' if b<0 else ''}{abs(b):.3f} ({se:.3f})"

def _style(sig):
    if sig in ('***','**','*'): return 'black', True
    if sig == 't':              return '#555555', False
    return '#999999', False

def _dr2(out, fm3, fm1):
    dr2  = fm3.loc[out,'R2'] - (fm1.loc[out,'R2'] if out in fm1.index else np.nan)
    fp   = fm3.loc[out,'F_p']
    star = '***' if fp<.001 else ('**' if fp<.01 else ('*' if fp<.05 else ('†' if fp<.10 else '')))
    return f'{dr2:.3f}{star}'

def _f(out, fm3):
    fp   = fm3.loc[out,'F_p']
    star = '***' if fp<.001 else ('**' if fp<.01 else ('*' if fp<.05 else ('†' if fp<.10 else '')))
    return f'{fm3.loc[out,"F"]:.2f}{star}'

def build_rows():
    m3  = df_full[df_full['Model']=='M3'].set_index(['Outcome','Predictor'])
    fm3 = df_fit[df_fit['Model']=='M3'].set_index('Outcome')
    fm1 = df_fit[df_fit['Model']=='M1'].set_index('Outcome')

    def lookup(out, col):
        key = (out, col)
        if key not in m3.index: return None, None, None, 'ns'
        r = m3.loc[key]
        return (float(r['B_std']) if not pd.isna(r['B_std']) else None,
                float(r['B'])     if not pd.isna(r['B'])     else None,
                float(r['SE'])    if not pd.isna(r['SE'])    else None,
                str(r['Sig']))

    rows = []; cur_sec = None
    for col, section, label, _ in PRED_CONFIG:
        if section and section != cur_sec:
            rows.append(('section', section)); cur_sec = section
        cells = []
        for o in OUTCOMES:
            b_std, b, se, sig = lookup(o, col)
            cells.append((_fmt_beta(b_std, sig), _fmt_bse(b, se), *_style(sig)))
        rows.append(('data', label, cells))

    rows.append(('rule',))
    for lbl, vals in [
        ('n',            [str(int(fm3.loc[o,'n'])) for o in OUTCOMES]),
        ('R²',           [f'{fm3.loc[o,"R2"]:.3f}' for o in OUTCOMES]),
        ('ΔR² (M1→M3)', [_dr2(o, fm3, fm1) for o in OUTCOMES]),
        ('F (model)',    [_f(o, fm3) for o in OUTCOMES]),
    ]:
        rows.append(('stat', lbl, vals))
    return rows


def draw(out_base='regression_table'):
    rows = build_rows()
    N = len(OUTCOMES)
    ROW_H=0.28; SEC_H=0.30; RULE_H=0.06; HEAD_H=0.50; FOOT_H=1.0; TOP_H=0.55
    FIG_W=14.0
    FIG_H = max(HEAD_H + sum(SEC_H if r[0]=='section' else
                             RULE_H if r[0]=='rule' else ROW_H for r in rows)
                + FOOT_H + TOP_H + 0.4, 7.0)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

    L=0.03; R=0.98; PF=0.22; DF=(1-PF)/(N*2)
    def cx(c): return L if c==0 else L+PF*(R-L)+(c-1)*DF*(R-L)
    def hl(y, lw, x0=None, x1=None):
        fig.add_artist(mlines.Line2D([x0 or L, x1 or R],[y,y],
            transform=fig.transFigure,color='#1c1c1c',linewidth=lw,clip_on=False))

    y = 0.97
    fig.text(L, y,
        'Table 1   OLS Regression of Ecological Satisfaction on '
        'Participation and Public Awareness (N=232)',
        transform=fig.transFigure, fontsize=10, fontweight='bold', va='top', ha='left')
    y -= TOP_H*0.55/FIG_H
    fig.text(L, y, 'Full model (M3); β* = standardised, B (SE) = unstandardised. '
        'Controlling for demographics.',
        transform=fig.transFigure, fontsize=8.5, style='italic', color='#444444', va='top', ha='left')
    y -= TOP_H*0.45/FIG_H

    hl(y, 1.5)
    y -= HEAD_H*0.46/FIG_H
    for i, out in enumerate(OUTCOMES):
        mid = (cx(1+i*2)+cx(3+i*2))/2
        fig.text(mid, y, out, transform=fig.transFigure,
                 fontsize=9, fontweight='bold', ha='center', va='center')
        hl(y-0.012, 0.5, cx(1+i*2)+0.002, cx(3+i*2)-0.002)

    y -= HEAD_H*0.54/FIG_H
    fig.text(L+0.002, y, 'Predictor', transform=fig.transFigure,
             fontsize=8.5, style='italic', va='center', ha='left')
    for i in range(N):
        fig.text((cx(1+i*2)+cx(2+i*2))/2, y, 'β*',
                 transform=fig.transFigure, fontsize=8.5, ha='center', va='center')
        fig.text((cx(2+i*2)+cx(3+i*2))/2, y, 'B (SE)',
                 transform=fig.transFigure, fontsize=8.5, ha='center', va='center')
    y -= 0.014; hl(y, 1.0)

    for row in rows:
        k = row[0]
        if k == 'section':
            y -= SEC_H/FIG_H*0.35
            hl(y+SEC_H/FIG_H*0.35-0.003, 0.4)
            fig.text(L+0.003, y, row[1], transform=fig.transFigure,
                     fontsize=8.5, style='italic', color='#444444', va='center', ha='left')
            y -= SEC_H/FIG_H*0.65
        elif k == 'data':
            _, label, cells = row
            fig.text(L+0.018, y, label, transform=fig.transFigure,
                     fontsize=8.5, va='center', ha='left')
            for i, (bs, bse, col, bold) in enumerate(cells):
                kw = dict(transform=fig.transFigure, fontsize=8, ha='center',
                          va='center', color=col, fontfamily='monospace',
                          fontweight='bold' if bold else 'normal')
                fig.text((cx(1+i*2)+cx(2+i*2))/2, y, bs, **kw)
                fig.text((cx(2+i*2)+cx(3+i*2))/2, y, bse, **{**kw,'fontsize':7.5})
            y -= ROW_H/FIG_H
        elif k == 'rule':
            hl(y+ROW_H/FIG_H*0.3, 0.8); y -= RULE_H/FIG_H
        elif k == 'stat':
            _, label, vals = row
            fig.text(L+0.003, y, label, transform=fig.transFigure,
                     fontsize=8.5, style='italic', color='#444444', va='center', ha='left')
            for i, v in enumerate(vals):
                fig.text((cx(1+i*2)+cx(3+i*2))/2, y, v,
                         transform=fig.transFigure, fontsize=8, ha='center', va='center')
            y -= ROW_H/FIG_H

    hl(y+ROW_H/FIG_H*0.3, 1.5)
    y -= 0.018
    notes = [
        'Note. EMQ = Environmental and ecological quality; EHB = Environmental hydrological benefits; '
        'ECO = Biodiversity satisfaction; MGT = Park management satisfaction; OVERALL = mean of all four constructs.',
        'β* = standardised coefficient; B = unstandardised; SE = standard error. '
        'All demographic variables pre-coded as ordinal integers.',
        'ΔR² = increment from M1 (demographics only) to M3 (full model). '
        'Listwise deletion per model.',
        '† p < .10   * p < .05   ** p < .01   *** p < .001',
    ]
    for line in notes:
        fig.text(L, y, line, transform=fig.transFigure,
                 fontsize=7.5, color='#555555', va='top', ha='left')
        y -= 0.030

    fig.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{out_base}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_base}.png / .pdf')


draw('regression_table')
print('Done.')

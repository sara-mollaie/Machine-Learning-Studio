"""Metadata-driven regression evaluation, using the existing evaluation theme."""
import math
from types import SimpleNamespace
from UserInterface.EvaluateModule import EvaluateModule, MX, PY, PH, LW, RX, RW, DARK_A, DARK_B
from UserInterface.Theme import theme

METRICS = ('MAE', 'MSE', 'RMSE', 'R2', 'R2 Percentage', 'Accuracy')
LABELS = ('MAE', 'MSE', 'RMSE', 'R²', 'R² %', 'Accuracy')

def number(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def fmt(v, key=''):
    v = number(v)
    if v is None: return '—'
    if key in ('Accuracy', 'R2 Percentage'): return f'{v:.2f}%'
    if abs(v) >= 10000 or 0 < abs(v) < .001: return f'{v:.2e}'
    return f'{v:.3f}' if key == 'R2' else f'{v:,.2f}'

class RegressionEvaluate(EvaluateModule):
    def __init__(self, master, scale=1.0, on_navigate=None):
        self.metric = 'Accuracy'
        super().__init__(master, scale, on_navigate)

    def _data(self):
        e = self.meta.get('model_evaluation') or {}
        rows = [r for r in e.get('results', []) if isinstance(r, dict) and r.get('Model')]
        rows.sort(key=lambda r: number(r.get('RMSE')) if number(r.get('RMSE')) is not None else math.inf)
        name = e.get('recommended_model')
        best = next((r for r in rows if r['Model'] == name), {})
        rule = str(e.get('selection_rule') or 'Selection rule unavailable')
        return SimpleNamespace(name=str(self.meta.get('name') or self.ds.name), rows=rows, best=best, model=name, rule=rule, ev=e)

    def _draw_back_button(self):
        S = self.s
        dark = theme.name == "dark"
        color = "#f1d5fa" if dark else "#ffffff"
        bg = "#50336f" if dark else "#713b91"
        edge = "#805ba1" if dark else "#713b91"
        w, h = 90, 30
        x, y = MX + 4, 103
        tag = "evaluate_back"
        self.create_image(S(x), S(y), image=self._shape(w, h, 10, bg, border=(edge, 255)), anchor="nw", tags=(tag,))
        cy = y + h / 2
        self.create_line(S(x+29), S(cy), S(x+14), S(cy), fill=color, width=max(1,S(2)), capstyle="round", tags=(tag,))
        self.create_line(S(x+21), S(cy-7), S(x+14), S(cy), S(x+21), S(cy+7), fill=color, width=max(1,S(2)), capstyle="round", joinstyle="round", tags=(tag,))
        self.text(x+37, cy, "Back", 16, color, True, tags=(tag,))
        self.hover(tag)
        self.tag_bind(tag, "<Button-1>", lambda e: self.on_navigate("predictor"))

    def _draw_page(self):

        c, S = theme.c, self.s
        d = self._data()

        # ----- مسیر + عنوان -----
        crumb_y = 118
        x = MX + 108
        self._draw_back_button()

        self.text(x, crumb_y, "Datasets", 20, c("accent"), True, tags=("crumb_ds",))
        self.hover("crumb_ds")
        self.tag_bind("crumb_ds", "<Button-1>", lambda e: self.on_navigate("datasets"))
        x += self.text_width("Datasets", 20, True) + 10

        self.text(x, crumb_y, "/", 20, c("body"))
        x += self.text_width("/", 20) + 10

        self.text(x, crumb_y, d.name, 20, c("body"), tags=("crumb_name",))
        self.hover("crumb_name")
        self.tag_bind("crumb_name", "<Button-1>", lambda e: self.on_navigate("predictor"))
        x += self.text_width(d.name, 20) + 10

        self.text(x, crumb_y, "/  Evaluate", 20, c("body"))

        # The heading is included in the background image.

        self._draw_stats(d)

        # ----- پنل‌ها -----
        left, lp = self._glass(LW, PH, 36, c("card_cols")[0], c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), c("card_border"))
        right, rp = self._glass(RW, PH, 36, DARK_A, DARK_B, 255,
                                c("shadow_a"), c("shadow_rgb"), None)
        self.create_image(S(MX) - lp, S(PY) - lp, image=left, anchor="nw")
        self.create_image(S(RX) - rp, S(PY) - rp, image=right, anchor="nw")

        self._draw_left(d)
        self._draw_right(d)

    def _draw_stats(self,d):
        c,S=theme.c,self.s
        for i,(kind,key,label) in enumerate(zip(('features','columns','clusters'),('Accuracy','RMSE','R2'),('Accuracy','RMSE','R² Score'))):
            x=MX+1544-3*268-28+i*282; y=112
            img,pad=self._glass(268,92,28,c('card_cols')[0],c('card_end'),c('card_alpha'),c('shadow_a'),c('shadow_rgb'),c('card_border'),10,6)
            self.create_image(S(x)-pad,S(y)-pad,image=img,anchor='nw')
            self.create_image(S(x+20),S(y+21),image=self._info_icon(kind,50),anchor='nw')
            self.text(x+86,y+36,fmt(d.best.get(key),key),28,c('title'),True)
            self.text(x+86,y+68,label,17,c('body'))

    def _select(self,key):
        self.metric=key
        self._build()

    def _draw_left(self,d):
        c,S=theme.c,self.s; x=MX+48; w=LW-96; right=x+w
        self.text(x,PY+44,'Model Comparison',30,c('title'),True)
        self.text(right,PY+44,f'{len(d.rows)} models',17,c('body'),anchor='e')
        for i,(key,label) in enumerate(zip(METRICS,LABELS)):
            xx=x+i*(w/6); bw=w/6-8; tag='metric_'+str(i); active=key==self.metric
            self.create_image(S(xx),S(PY+75),image=self._shape(bw,34,12,c('accent') if active else self._track()[0],alpha=65 if active else 255),anchor='nw',tags=(tag,))
            self.text(xx+bw/2,PY+92,label,16,c('title'),True,anchor='center',tags=(tag,))
            self.hover(tag); self.tag_bind(tag,'<Button-1>',lambda e,k=key:self._select(k))
        lower=self.metric in METRICS[:3]
        self.text(x,PY+132,'Lower is better' if lower else 'Higher is better',16,c('body'))
        rows=[r for r in d.rows if number(r.get(self.metric)) is not None]
        rows.sort(key=lambda r:number(r[self.metric]),reverse=not lower)
        vals=[number(r[self.metric]) for r in rows]; lo=min([0]+vals); hi=max([0]+vals); span=hi-lo or 1
        bx=x+230; bw=w-350; zero=bx+bw*(0-lo)/span
        for i,r in enumerate(rows[:5]):
            y=PY+166+i*40; v=number(r[self.metric]); selected=r['Model']==d.model
            self.text(x,y,str(r['Model']),17,c('title'),selected)
            self.create_image(S(bx),S(y-6),image=self._shape(bw,12,6,self._track()[0]),anchor='nw')
            end=bx+bw*(v-lo)/span
            if abs(end-zero)>1:
                self.create_image(S(min(end,zero)),S(y-6),image=self._shape(abs(end-zero),12,6,'#c04b91' if selected else '#a58be0'),anchor='nw')
            self.text(right,y,fmt(v,self.metric),21,c('title'),True,anchor='e')
        self.text(x,PY+388,'Evaluation Metrics',23,c('title'),True)
        self.text(right,PY+388,'Pink = recommended',15,c('accent'),anchor='e')
        xs=[x+300+i*(w-300)/5 for i in range(6)]
        self.text(x,PY+424,'Model',15,c('body'),True)
        for xx,label in zip(xs,LABELS): self.text(xx,PY+424,label,17,c('body'),True,anchor='e')
        for i,r in enumerate(d.rows[:5]):
            y=PY+460+i*36; sel=r['Model']==d.model
            if sel: self.create_image(S(x-8),S(y-15),image=self._shape(w+16,30,10,c('accent'),alpha=35),anchor='nw')
            self.text(x,y,str(r['Model']),17,c('title'),sel)
            for xx,key in zip(xs,METRICS): self.text(xx,y,fmt(r.get(key),key),18,c('title'),True,anchor='e')
        if not d.rows: self.text(x,PY+220,'No saved model metrics available.',19,c('body'))

    def _draw_right(self,d):
        S=self.s; x=RX+48; w=RW-96; right=x+w
        muted='#d8cbe5'; accent='#ffb5d9'
        metric_colors=('#e1caff','#e1caff','#e1caff','#b8d9ff','#b8d9ff','#a6e9cf')
        def text(y,t,size=19,b=False,color='white',xx=x):
            self.text(xx,PY+y,t,size,color,b)
        def card(y,h,xx=x,ww=w):
            self.create_image(S(xx),S(PY+y),image=self._shape(ww,h,18,'#ffffff',alpha=30),anchor='nw')
        text(35,'Model Performance',30,True)
        card(65,125)
        text(87,'Recommended Model',19,False,muted,x+18)
        model=str(d.model or 'Unavailable')
        size=30
        while size>18 and self.text_width(model,size,True)>w-36:
            size-=1
        text(120,model,size,True,accent,x+18)
        for i,line in enumerate(self._wrap(d.rule,16,False,w-36,2)):
            text(151+i*19,line,16,False,muted,x+18)
        descriptions=('Mean Absolute Error','Mean Squared Error','Root Mean Squared Error','Coefficient of Determination','R² × 100','Custom RMSE-based score')
        for i,(key,label,desc) in enumerate(zip(METRICS,LABELS,descriptions)):
            ww=(w-18)/2; xx=x+(i%2)*(ww+18); yy=199+(i//2)*112
            card(yy,100,xx,ww)
            text(yy+19,label,19,True,'#e0cbed',xx+16)
            value=fmt(d.best.get(key),key)
            size=32
            while size>18 and self.text_width(value,size,True)>ww-32:
                size-=1
            text(yy+53,value,size,True,metric_colors[i],xx+16)
            for j,line in enumerate(self._wrap(desc,15,False,ww-32,2)):
                text(yy+77+j*16,line,15,False,muted,xx+16)
        # Compact insight, based only on saved evaluation metrics.
        rmse = number(d.best.get('RMSE'))
        r2 = number(d.best.get('R2 Percentage'))
        if r2 is None:
            raw_r2 = number(d.best.get('R2'))
            r2 = raw_r2 * 100 if raw_r2 is not None else None
        competitors = [number(r.get('RMSE')) for r in d.rows if r['Model'] != d.model]
        competitors = [v for v in competitors if v is not None and v >= 0]
        parts = []
        if rmse is not None and competitors:
            other = min(competitors)
            if other > 0 and rmse < other:
                parts.append(f'RMSE is {(other-rmse)/other*100:.2f}% lower than the next-best model.')
            else:
                parts.append(f'Selected model RMSE: {fmt(rmse)}; best competing RMSE: {fmt(other)}.')
        elif rmse is not None:
            parts.append(f'Selected model RMSE: {fmt(rmse)}.')
        if r2 is not None:
            parts.append(f'R²: {r2:.2f}% of target variance explained.' if 0 <= r2 <= 100 else f'R² score: {r2/100:.3f}.')
        insight = ' '.join(parts) or 'Evaluation metrics are not available.'
        iy = 550
        self.create_image(S(x),S(PY+iy),image=self._shape(w,90,18,'#ffffff',alpha=30,border=('#795184',255)),anchor='nw')
        text(iy+17,'INSIGHT',16,True,accent,x+18)
        size = 16
        for j,line in enumerate(self._wrap(insight,size,False,w-36,4)):
            text(iy+36+j*15,line,size,False,'#eee3f6',x+18)

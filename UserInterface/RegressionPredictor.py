"""Regression UI reusing the original Clustering canvas and drawing helpers."""
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
from UserInterface.HomeModule import BASE
from tkinter import messagebox
import tkinter as tk
from calendar import month_name
from UserInterface.Theme import FONT
from UserInterface.PredictorModule import PredictorModule, MX, PY, PH, LW, RX, RW, BAR_TRACK, BAR_ON
from UserInterface.Theme import theme
from .RegressionData import RegressionData


class RegressionPredictor(PredictorModule):
    PAGE_SIZE = 6
    DATA_CLASS = RegressionData          # <-- CHANGE 1: subclasses (Classification) override this

    def __init__(self, master, scale=1.0, on_navigate=None):
        self.data = None
        self._importance_pool = ThreadPoolExecutor(max_workers=1)
        self._importance_job = None
        self._importance_data = None
        self._importance_status = None
        self.feature_page = 0
        self.manual = set()
        self._show_missing = False
        super().__init__(master, scale=scale, on_navigate=on_navigate)

    def _load_pred_bg(self, name):
        folder = BASE / "assets" / "page_images"
        path = folder / f"predict Reg_background_{name}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Regression background not found: {path}")
        with Image.open(path) as image:
            return image.convert("RGB").resize((self.W, self.H), Image.Resampling.LANCZOS)

    def _bg_photo(self):
        name = theme.name
        key = ("regression_background", name, self.W, self.H)
        if key not in self._bgs:
            image = self._load_pred_bg(name)
            self._bgs[key] = ImageTk.PhotoImage(image, master=self)
        return self._bgs[key]

    def preload(self, datasets):
        pass

    def load_dataset(self, ds):
        if self._importance_job is not None:
            self._importance_job.cancel()
        self._importance_job = None
        self._importance_data = None
        self._importance_status = None
        self.ds, self.data, self.bundle = ds, None, None
        self.result = self.notice = self.error = None
        self.feature_page = 0
        self.manual = set()
        self._show_missing = False
        self.vals = {}
        self._date_years = {}
        self._date_months = {}
        try:
            self.data = self.DATA_CLASS(ds)          # <-- CHANGE 2: was RegressionData(ds)
            self.meta = self.data.meta
            ranges, decimals = {}, {}
            integer_features = set()
            self._numeric_choices = {}
            if self.DATA_CLASS is not RegressionData:
                # Inspect actual processed values, not just integer min/max.
                import pandas as pd
                processed = self.meta.get('output_files', {}).get('processed_dataset')
                if isinstance(processed, str):
                    source = Path(ds.path).parent.parent / processed.replace('\\', '/')
                    if source.is_file():
                        try:
                            frame = pd.read_csv(source, usecols=lambda c: c in self.data.features)
                            for col in frame:
                                raw = frame[col].dropna()
                                nums = pd.to_numeric(raw, errors='coerce')
                                if len(nums) and nums.notna().all() and all(math.isfinite(v) for v in nums):
                                    unique = sorted(set(float(v) for v in nums))
                                    if 2 <= len(unique) <= 10 and col.lower().strip() not in ('age', 'sitting hours', 'age at heart attack'):
                                        self._numeric_choices[col] = unique
                                if len(nums) and nums.notna().all() and all(
                                    math.isfinite(v) and float(v).is_integer() for v in nums
                                ):
                                    integer_features.add(col)
                        except (OSError, ValueError):
                            pass
                # Verified integer fields in Blood Transfusion, even without CSV.
                if {'Recency', 'Frequency', 'Monetary', 'Time'} == set(self.data.features):
                    integer_features.update(self.data.features)
            if self.DATA_CLASS is not RegressionData:
                for f in self.data.features:
                    key = f.lower().strip().replace('_', ' ')
                    if key == 'pericardial effusion':
                        self._numeric_choices.setdefault(f, [0, 1])
            for f in self.data.features:
                options = self.data.categories.get(f)
                if f.lower() != 'x1 transaction date' and options is not None and len(options):
                    try:
                        if len(options)>10 and all(math.isfinite(float(v)) for v in options):
                            options = sorted(options,key=float)
                            self.data.categories[f] = options
                    except (ValueError,TypeError):
                        pass
                    ranges[f] = (0, len(options)-1)
                    decimals[f] = 0
                    continue
                st = self.data.stats.get(f, {})
                try:
                    lo, hi = float(st['min']), float(st['max'])
                    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                        raise ValueError('Invalid range')
                    if f.lower() != 'x1 transaction date' and f in self.data.categorical:
                        raise ValueError('Unknown categories')
                    ranges[f] = (lo, hi)
                    dataset_name = str(self.data.bundle.get('dataset_name', '')).strip().casefold()
                    feature_name = f.lower().strip().replace('_', ' ')
                    if 'fertility' in dataset_name and feature_name in ('age', 'sitting hours'):
                        decimals[f] = 2
                    elif 'echocardiogram' in dataset_name and feature_name == 'age at heart attack':
                        decimals[f] = 0
                    elif f in integer_features:
                        decimals[f] = 0
                    else:
                        decimals[f] = 0 if f.lower() in ('weight','age','x4 number of convenience stores') or 'volume' in f.lower() else 4 if hi-lo < 1 else 3
                except (KeyError, TypeError, ValueError):
                    ranges[f] = (0, 0)
                    self.manual.add(f)
            self.bundle = dict(features=list(self.data.features), range=ranges,
                               decimals=decimals, labels={f:self._pretty(f) for f in ranges})
            for f in self.data.features:
                if self._kind(f) == 'slider' and self._choices(f):
                    self.bundle['decimals'][f] = 0
            self.vals = dict.fromkeys(self.data.features)
            self._start_importance(ds)
        except Exception as exc:
            self.error = f'Could not load the model files: {exc}'
        self._dirty = True
        if self.winfo_ismapped():
            self._build()

    def _date_options(self, f):
        lo, hi = self.bundle['range'][f]
        if f in self.manual or not (1 <= lo <= hi <= 9999):
            return {}
        # Metadata statistics are rounded to two decimal places.
        return {year: months for year in range(max(1, int(lo)-1), min(9999, int(hi))+1)
                if (months := [month for month in range(1, 13)
                    if lo-.00501 <= round(year+month/12, 3) <= hi+.00501])}

    def _choose_date(self, f, part, event):
        options = self._date_options(f)
        menu = tk.Menu(self, tearoff=False, font=(FONT, -self.s(17)),
                       bg=theme.c('dlg_bg'), fg=theme.c('title'),
                       activebackground='#873c78', activeforeground='white')
        def choose_year(year):
            self._date_years[f] = year
            self._date_months.pop(f, None)
            self._set_feature(f, None)
        def choose_month(month):
            year = self._date_years[f]
            self._date_months[f] = month
            self._set_feature(f, round(year+month/12, 3))
        if part == 'year' or f not in self._date_years:
            for year in options:
                menu.add_command(label=str(year), command=lambda y=year: choose_year(y))
        else:
            for month in options.get(self._date_years[f], []):
                menu.add_command(label=month_name[month], command=lambda m=month: choose_month(m))
        if menu.index('end') is None:
            menu.add_command(label='Date range unavailable', state='disabled')
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _fmt_feat(self, f, v):
        options = self.data.categories.get(f)
        if options is not None and len(options):
            return str(options[int(v)])
        if f in getattr(self, '_numeric_choices', {}):
            return format(float(v), 'g')
        if f in self.manual or self._kind(f) != 'slider':
            return str(v)
        return super()._fmt_feat(f, v)

    def _choices(self, f):
        options = self.data.categories.get(f)
        if options is not None and len(options):
            return [(str(v), i) for i, v in enumerate(options)]
        numeric = getattr(self, '_numeric_choices', {}).get(f)
        if numeric and f not in self.manual:
            return [(format(v, 'g'), v) for v in numeric]
        st = self.data.stats.get(f, {})
        key = f.lower().strip()
        try:
            lo, hi = float(st['min']), float(st['max'])
        except (KeyError, ValueError, TypeError):
            return []
        if key in ('year', 'month', 'day', 'hour', 'hr', 'cylinders', 'model year', 'origin', 'week of month', 'day of week', 'weekday', 'season', 'orientation', 'glazing area distribution') or (lo == 0 and hi == 1 and key.startswith(('is_', 'weekday_is_', 'data_channel_is_'))):
            if lo.is_integer() and hi.is_integer() and 0 <= hi-lo <= 120:
                return [(str(i), i) for i in range(int(lo), int(hi)+1)]
        return []

    def _kind(self, f):
        if f.lower() == 'x1 transaction date':
            return 'date'
        choices = self._choices(f)
        if choices:
            if len(choices) > 10:
                try:
                    if all(math.isfinite(float(label)) for label, _ in choices):
                        return 'slider'
                except (ValueError, TypeError):
                    pass
                return 'scroll_choice'
            return 'choice'
        lo, hi = self.bundle['range'][f]
        if f in self.manual:
            return 'entry'
        return 'slider'

    def _draw_sliders(self, px, pw):
        from UserInterface.PredictorModule import FILL_A, FILL_B
        c, S = theme.c, self.s
        ui = self._ui()
        all_features = self.bundle['features']
        count = max(1, math.ceil(len(all_features)/self.PAGE_SIZE))
        self.feature_page = min(self.feature_page, count-1)
        start = self.feature_page*self.PAGE_SIZE
        visible = all_features[start:start+self.PAGE_SIZE]
        self._sl = {}
        x0, inner = px+48, pw-96
        dark = theme.name == 'dark'
        self._section_icon(x0, PY+36, 'features', '#f3a4d0' if dark else '#a3306f', 32)
        self.text(x0+48, PY+52, 'Features', 28, c('title'), True)
        self._counter = self.create_text(S(px+pw-48), S(PY+52), text='', anchor='e', fill=c('body'), font=(FONT,-S(17)))
        self._update_counter()
        gap = 40
        w = (inner-gap)/2
        pitch = 100 if count>1 else 108
        full = self._pill_pil(w,14,FILL_A,FILL_B)
        track = self._shape(w,14,7,ui['track'],border=(ui['track_border'],255))
        hit, knob = self._hit(w,34), self._knob_img()
        for i,f in enumerate(visible):
            row,col = divmod(i,2)
            x,y = x0+col*(w+gap), PY+108+row*pitch
            ty = y+30
            lo,hi = self.bundle['range'][f]
            v = self.vals[f]
            kind = self._kind(f)
            self.create_image(S(x-10),S(y-19),image=self._shape(w+20,88,15,'#ffffff',alpha=18 if dark else 65,border=(ui['track_border'],150)),anchor='nw')
            self.text(x,y,self._fit(self.bundle['labels'][f],19,True,w-90 if kind=='slider' else w),19,c('title'),True)
            if kind == 'slider':
                t = 0.0 if v is None or hi<=lo else max(0.0,min(1.0,(v-lo)/(hi-lo)))
                tag = f'sl_{i}'
                state = 'hidden' if v is None else 'normal'
                val_id = self.create_text(S(x+w),S(y),text='—' if v is None else self._fmt_feat(f,v),anchor='e',fill=c('body') if v is None else c('title'),font=(FONT,-S(19),'bold'))
                self.create_image(S(x+w/2),S(ty),image=hit,anchor='center',tags=(tag,))
                self.create_image(S(x),S(ty),image=track,anchor='w',tags=(tag,))
                photo = ImageTk.PhotoImage(full.crop((0,0,max(1,round(t*full.width)),full.height)))
                fill_id = self.create_image(S(x),S(ty),image=photo,anchor='w',tags=(tag,),state=state)
                knob_id = self.create_image(S(x+t*w),S(ty),image=knob,anchor='center',tags=(tag,),state=state)
                self._sl[f] = dict(x=x,w=w,ty=ty,lo=lo,hi=hi,val=val_id,fill=fill_id,knob=knob_id,photo=photo,full=full)
                color = '#e1c9fa' if dark else '#58377e'
                self.text(x,ty+23,self._fmt_feat(f,lo),16,color,True)
                self.text(x+w,ty+23,self._fmt_feat(f,hi),16,color,True,anchor='e')
                self.hover(tag)
                self.tag_bind(tag,'<Button-1>',lambda e,name=f:self._drag(name,e))
                self.tag_bind(tag,'<B1-Motion>',lambda e,name=f:self._drag(name,e))
            elif kind == 'date':
                ty = y+30
                self._sl[f] = dict(x=x,w=w,ty=ty,lo=lo,hi=hi)
                border = ('#7a5aa6' if dark else '#bfaedf',255)
                half = (w-10)/2
                year = self._date_years.get(f)
                month = self._date_months.get(f)
                for part, xx, label in [('year', x, str(year) if year else 'Select year'),
                                        ('month', x+half+10, month_name[month] if month else 'Select month')]:
                    tag = f'date_{i}_{part}'
                    self.create_image(S(xx),S(ty-16),image=self._shape(half,32,12,self._pal()['search_bg'],border=border),anchor='nw',tags=(tag,))
                    self.text(xx+10,ty,self._fit(label,17,False,half-34),17,c('title'),tags=(tag,))
                    self.text(xx+half-10,ty,'▾',16,c('title'),anchor='e',tags=(tag,))
                    self.hover(tag)
                    self.tag_bind(tag,'<Button-1>',lambda e,name=f,p=part:self._choose_date(name,p,e))
                hint = 'Gregorian date: year, then month' if v is None else f'{month_name[month]} {year}'
                self.text(x,ty+25,hint,13,c('body'))
            else:
                ty = y+35
                self._sl[f] = dict(x=x,w=w,ty=ty,lo=lo,hi=hi)
                label = ('Select value' if kind in ('choice','scroll_choice') else 'YYYY / MM' if kind=='date' else 'Enter value') if v is None else self._fmt_feat(f,v)
                tag = f'input_{i}'
                border = ('#7a5aa6' if dark else '#bfaedf',255)
                self.create_image(S(x),S(ty-16),image=self._shape(w,32,12,self._pal()['search_bg'],border=border),anchor='nw',tags=(tag,))
                self.text(x+12,ty,self._fit(label,19,False,w-44),19,c('title'),tags=(tag,))
                self.text(x+w-12,ty,'▾' if kind in ('choice','scroll_choice') else '…',18,c('title'),anchor='e',tags=(tag,))
                self.hover(tag)
                self.tag_bind(tag,'<Button-1>',lambda e,name=f:self._edit_feature(name,e))
        self._highlight_missing()
        if count>1:
            y = PY+420
            for label,x,delta in [('Previous',x0,-1),('Next',x0+inner-60,1)]:
                if (delta == -1 and self.feature_page == 0) or (delta == 1 and self.feature_page == count - 1):
                    continue
                tag = 'feature_page_'+str(delta)
                self.text(x,y,label,15,c('accent'),True,tags=(tag,))
                self.hover(tag)
                self.tag_bind(tag,'<Button-1>',lambda e,d=delta:self._change_page(d))
            self.text(x0+inner/2,y,f'{self.feature_page+1} / {count}',15,c('body'),anchor='center')

    def _highlight_missing(self):
        self.delete('missing_outline')
        if not self._show_missing:
            return
        S = self.s
        violet = '#c49aff' if theme.name == 'dark' else '#a879ed'
        for f, slot in self._sl.items():
            if self.vals.get(f) is not None:
                continue
            x, w, y = slot['x'], slot['w'], slot['ty']
            h = 10 if self._kind(f) == 'slider' else 18
            # Outline only: leave the control interior and bindings intact.
            r = min(8, h)
            l, t, rr, b = x-2, y-h, x+w+2, y+h
            pts = [(l+r,t),(rr-r,t),(rr,t),(rr,t+r),(rr,b-r),(rr,b),
                   (rr-r,b),(l+r,b),(l,b),(l,b-r),(l,t+r),(l,t),(l+r,t)]
            tag = 'missing_outline'
            item = self.create_polygon(*[S(v) for point in pts for v in point],
                smooth=True, fill='', outline=violet, width=max(2,S(2.5)), tags=(tag,))
            if self._kind(f) == 'slider':
                self.tag_bind(item,'<Button-1>',lambda e,name=f:self._drag(name,e))
                self.tag_bind(item,'<B1-Motion>',lambda e,name=f:self._drag(name,e))
            else:
                self.tag_bind(item,'<Button-1>',lambda e,name=f:self._edit_feature(name,e))

    def _refresh_missing_notice(self):
        if self._show_missing:
            count = sum(v is None for v in self.vals.values())
            self._show_missing = bool(count)
            self.notice = f'Set every feature value ({count} left).' if count else None

    def _set_feature(self, f, value):
        self.vals[f] = value
        self.result = self.notice = None
        self._refresh_missing_notice()
        self._build()

    def _edit_feature(self, f, event):
        kind = self._kind(f)
        if kind == 'date':
            self._choose_date(f, 'year' if f not in self._date_years else 'month', event)
            return
        if kind == 'scroll_choice':
            popup = tk.Toplevel(self)
            popup.transient(self.winfo_toplevel())
            popup.title(self._pretty(f))
            popup.geometry(f'+{event.x_root}+{max(0,min(event.y_root,self.winfo_screenheight()-self.s(320)))}')
            bg = theme.c('dlg_bg')
            popup.configure(bg=bg)
            bar = tk.Scrollbar(popup)
            bar.pack(side='right',fill='y')
            choices = self._choices(f)
            items = tk.Listbox(popup,height=10,font=(FONT,-self.s(19)),bg=bg,fg=theme.c('title'),selectbackground='#873c78',selectforeground='white',yscrollcommand=bar.set,exportselection=False,relief='flat')
            items.pack(side='left',fill='both',expand=True)
            bar.config(command=items.yview)
            for label,_ in choices:
                items.insert('end',label)
            def choose(e=None):
                selection = items.curselection()
                if selection:
                    value = choices[selection[0]][1]
                    popup.destroy()
                    self._set_feature(f,value)
            items.bind('<ButtonRelease-1>',choose)
            items.bind('<Return>',choose)
            popup.bind('<Escape>',lambda e:popup.destroy())
            items.focus_set()
            return
        if kind == 'choice':
            menu = tk.Menu(self, tearoff=False, font=(FONT, -self.s(19)),
                           bg=theme.c('dlg_bg'), fg=theme.c('title'),
                           activebackground='#873c78', activeforeground='white',
                           borderwidth=0, relief='flat')
            for label, value in self._choices(f):
                menu.add_command(label=label, command=lambda v=value:self._set_feature(f,v))
            try:
                menu.tk_popup(event.x_root,event.y_root)
            finally:
                menu.grab_release()
            return
        slot = self._sl[f]
        if getattr(self, '_inline_entry', None) is not None:
            self._inline_entry.destroy()
        entry = tk.Entry(self, font=(FONT,-self.s(19)), bg='#35214d' if theme.name=='dark' else '#f3edf9',
                         fg=theme.c('title'), insertbackground=theme.c('title'), relief='flat')
        self._inline_entry = entry
        entry.place(x=self.s(slot['x']+10), y=self.s(slot['ty']-15),
                    width=self.s(slot['w']-20), height=self.s(30))
        if self.vals[f] is not None:
            entry.insert(0,str(self.vals[f]))
        entry.select_range(0,'end')
        entry.focus_set()
        def commit(e=None):
            try:
                raw = entry.get().strip()
                value = raw if f in self.data.categorical else float(raw)
                if not raw or isinstance(value,float) and not math.isfinite(value):
                    raise ValueError()
                if f not in self.manual:
                    lo,hi = self.bundle['range'][f]
                    if not lo <= value <= hi:
                        raise ValueError()
            except ValueError:
                entry.configure(highlightthickness=1,highlightbackground='#a879ed',highlightcolor='#a879ed')
                return
            entry.destroy()
            self._inline_entry = None
            self._set_feature(f,value)
        entry.bind('<Return>',commit)
        entry.bind('<Escape>',lambda e:entry.destroy())

    def _build(self):
        entry = getattr(self,'_inline_entry',None)
        if entry is not None:
            entry.destroy()
            self._inline_entry = None
        super()._build()

    def _change_page(self, delta):
        last = math.ceil(len(self.vals)/self.PAGE_SIZE)-1
        self.feature_page = max(0, min(last, self.feature_page+delta))
        self._build()

    def _drag(self, name, event):
        if name not in self.manual:
            self.result = self.notice = None
            super()._drag(name,event)
            self._refresh_missing_notice()
            self._highlight_missing()
            self._redraw_result()

    def _draw_algorithm(self, px, pw, pal):
        c, S = theme.c, self.s
        x, inner = px+48, pw-96
        top = PY+PH-168
        aw = inner-170
        border = ('#7a5aa6' if theme.name=='dark' else '#bfaedf', 255)
        self.create_line(S(x), S(top-52), S(x+inner), S(top-52), fill=pal['sep'], width=max(1,S(1.5)))
        self._section_icon(x, top-39, 'algorithm', '#f3a4d0' if theme.name=='dark' else '#a3306f', 30)
        self.text(x+42, top-24, 'Algorithm', 20, c('title'), True)
        self.text(x+aw+20, top-24, 'Accuracy', 20, c('title'), True)
        best = self.data.best
        for xx,w,label in [(x,aw,'Regression'),(x+aw+20,150,self._metric('Accuracy', '%'))]:
            self.create_image(S(xx),S(top),image=self._shape(w,52,26,pal['search_bg'],border=border),anchor='nw')
            self.text(xx+26,top+26,self._fit(label,22,True,w-40),22,c('title'),True)

    def _metric(self, key, suffix=''):
        try:
            v=float(self.data.best[key])
            return f'{v:,.2f}{suffix}' if math.isfinite(v) else '—'
        except (KeyError, TypeError, ValueError):
            return '—'

    def _predict(self):
        if not self.bundle:
            return
        missing = sum(v is None for v in self.vals.values())
        self.result = None
        if missing:
            self._show_missing = True
            self.notice = f'Set every feature value ({missing} left).'
            first = next(i for i, f in enumerate(self.bundle['features']) if self.vals[f] is None)
            self.feature_page = first // self.PAGE_SIZE
            self._build()
            return
        else:
            self._show_missing = False
            self._highlight_missing()
            try:
                values = {f:self.data.categories[f][int(v)] if self._kind(f) != 'date' and f in self.data.categories and len(self.data.categories[f]) else v for f,v in self.vals.items()}
                self.result = self.data.predict(values)
                self.notice = None
            except Exception as exc:
                self.notice = str(exc)
        self._redraw_result()

    def _reset(self):
        self.feature_page = 0
        self._show_missing = False
        self.vals = dict.fromkeys(self.vals)
        self._date_years.clear()
        self._date_months.clear()
        self.result = self.notice = None
        self._build()

    @staticmethod
    def _compute_importance(data, metadata_path):
        from UserInterface.RegressionFeatureRanking import ranked_coefficients
        ranked = ranked_coefficients(data.meta)[:3]
        return [(name, abs(value)) for name, value in ranked], 'Linear Regression coefficients' if ranked else 'Feature coefficients unavailable'

    def _start_importance(self, ds):
        self._importance_status = 'Calculating feature importance...'
        job = self._importance_pool.submit(self._compute_importance, self.data, ds.path)
        self._importance_job = job
        def poll():
            if self._importance_job is not job:
                return
            if not job.done():
                self.after(150, poll)
                return
            try:
                self._importance_data, self._importance_status = job.result()
            except Exception:
                self._importance_data = []
                self._importance_status = 'Feature importance unavailable'
            if self.winfo_ismapped() and self.bundle:
                self._redraw_result()
        self.after(150, poll)

    def _draw_result(self):
        if not self.bundle:
            return
        S,T = self.s,('res',)
        left,right = RX+48,RX+RW-48
        width = right-left
        center = (left+right)/2
        muted, accent = '#e1c9ed', '#ffc0df'
        def txt(x,y,value,size=18,bold=False,color='white',anchor='w'):
            self.text(x,PY+y,str(value),size,color,bold,anchor=anchor,tags=T)
        def card(x,y,w,h):
            self.create_image(S(x),S(PY+y),image=self._shape(w,h,18,'#ffffff',alpha=28),anchor='nw',tags=T)
        def line(points,color,weight=2):
            self.create_line(*[S(v) for x,y in points for v in (x,PY+y)],fill=color,width=max(1,S(weight)),tags=T)
        def icon(x,y,kind):
            if kind == 'prediction':
                line([(x-10,y+9),(x-10,y-9)],accent,2)
                line([(x-10,y+9),(x+11,y+9)],accent,2)
                line([(x-6,y+3),(x,y-3),(x+4,y),(x+11,y-9)],accent,2)
            elif kind == 'award':
                self.create_oval(S(x-7),S(PY+y-10),S(x+7),S(PY+y+4),outline=accent,width=max(1,S(2)),tags=T)
                line([(x-5,y+3),(x-7,y+11),(x,y+7),(x+7,y+11),(x+5,y+3)],accent,2)
            else:
                for dy in (-6,0,6):
                    line([(x-10,y+dy),(x,y+dy-5),(x+10,y+dy),(x,y+dy+5),(x-10,y+dy)],accent,1.5)
        self._section_icon(left, PY+23, 'prediction', '#f3a4d0', 36, tags=T)
        txt(left+48,41,'Prediction',28,True)
        nodes=[(center-45,70),(center-45,84),(center-45,98),(center,77),(center,91),(center+45,84)]
        for x,y in nodes:
            for xx,yy in ([(center,77),(center,91)] if x<center else [(center+45,84)] if x==center else []):
                line([(x,y),(xx,yy)],'#b77bad',1.5)
        for x,y in nodes:
            r=4 if x!=center+45 else 6
            self.create_oval(S(x-r),S(PY+y-r),S(x+r),S(PY+y+r),fill=accent,outline='',tags=T)
        txt(center,120,self._fit('Predicted '+str(self.data.target),19,False,width),19,False,muted,'center')
        decimals = 0 if str(self.data.bundle.get('dataset_name', '')).strip().casefold() == 'abalone' else 5
        value = '—' if self.result is None else f'{self.result:,.{decimals}f}'
        value_size = 48
        while value_size > 1 and self.text_width(value, value_size, True) > width:
            value_size -= 1
        txt(center,163,value,value_size,True,accent,'center')
        msg=self.notice or ('Set every feature value, then press Predict.' if self.result is None else 'Prediction generated')
        txt(center,203,self._fit(msg,17,False,width),17,False,muted,'center')
        line([(left,224),(right,224)],'#795184',1)
        model=str(self.data.best['Model'])
        names={'SVR':'Support Vector Regression','Random Forest':'Random Forest Regressor','Gradient Boosting':'Gradient Boosting Regressor','Ridge':'Ridge Regression','Linear Regression':'Linear Regression'}
        card(left,238,width,83)
        icon(left+25,257,'award')
        txt(left+43,257,'Recommended Model',17,False,muted)
        txt(left+16,284,self._fit(model,24,True,width-32),24,True)
        full=names.get(model,'')
        if full and full!=model:
            txt(left+16,306,self._fit(full,16,False,width-32),16,False,muted)
        card(left,333,width,61)
        txt(left+16,352,'RMSE',19,True)
        txt(right-16,352,self._metric('RMSE'),24,True,accent,'e')
        txt(left+16,377,'Root Mean Squared Error',16,False,muted)
        card(left,408,width,187)
        icon(left+25,429,'layers')
        txt(left+46,429,'Key Predictors',21,True)
        ranked=self._importance_data or []
        if ranked:
            peak=max(abs(v) for _,v in ranked) or 1.0
            for i,(feature,importance) in enumerate(ranked[:3]):
                y=459+i*40
                txt(left+18,y,f'{i+1:02d}',15,True,accent)
                txt(left+48,y,self._fit(self._pretty(feature),17,True,width-70),17,True)
                bx,ex=left+48,right-20
                by=y+15
                self.create_line(S(bx),S(PY+by),S(ex),S(PY+by),fill='#795184',width=max(1,S(6)),capstyle=tk.ROUND,tags=T)
                self.create_line(S(bx),S(PY+by),S(bx+(ex-bx)*importance/peak),S(PY+by),fill=accent,width=max(1,S(6)),capstyle=tk.ROUND,tags=T)
            txt(left+18,581,self._fit(self._importance_status,13,False,width-36),13,False,muted)
        else:
            txt(center,505,self._fit(self._importance_status or 'Feature importance unavailable',17,False,width-32),17,False,muted,'center')
        line([(left,610),(right,610)],'#795184',1)
        txt(left,632,'Learning Type',18,True)
        txt(right,632,'Supervised Learning',17,True,muted,'e')


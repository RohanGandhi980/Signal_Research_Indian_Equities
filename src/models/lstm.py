from dataclasses import dataclass
import json, pathlib, numpy as np, pandas as pd, tensorflow as tf

@dataclass
class LSTMConfig:
    window:int=60; units:int=64; dropout:float=0.2
    lr:float=1e-3; epochs:int=25; batch_size:int=64

class LSTMForecaster:
    def __init__(self, cfg:LSTMConfig=LSTMConfig()):
        self.cfg=cfg; self.model=self._build()
        self.min_=None; self.max_=None

    def _build(self):
        i=tf.keras.Input(shape=(self.cfg.window,1))
        x=tf.keras.layers.LSTM(self.cfg.units)(i)
        x=tf.keras.layers.Dropout(self.cfg.dropout)(x)
        o=tf.keras.layers.Dense(1)(x)
        m=tf.keras.Model(i,o)
        m.compile(optimizer=tf.keras.optimizers.Adam(self.cfg.lr), loss="mse")
        return m

    def _scale(self, y):
        v=y.values.astype(float).reshape(-1,1)
        if self.min_ is None:
            self.min_,self.max_=float(v.min()),float(v.max())
        return (v-self.min_)/(self.max_-self.min_+1e-9)

    def _windows(self, y):
        s=self._scale(y)
        X,Y=[],[]
        for i in range(self.cfg.window,len(s)):
            X.append(s[i-self.cfg.window:i]); Y.append(s[i])
        return np.array(X),np.array(Y)

    def fit(self, y):
        X,Y=self._windows(y)
        self.model.fit(X,Y,epochs=self.cfg.epochs,batch_size=self.cfg.batch_size,verbose=0)
        return self

    def predict_next(self, y):
        s=self._scale(y)
        x=s[-self.cfg.window:].reshape(1,self.cfg.window,1)
        yhat_s=self.model.predict(x,verbose=0)[0,0]
        return float(yhat_s*(self.max_-self.min_)+self.min_)

    def save(self,path):
        self.model.save(path)
        meta={"min":self.min_,"max":self.max_,"window":self.cfg.window}
        pathlib.Path(path).with_suffix(".json").write_text(json.dumps(meta))

    @staticmethod
    def load(path:str):
        m=tf.keras.models.load_model(path)
        meta=json.loads(pathlib.Path(path).with_suffix(".json").read_text())
        obj=LSTMForecaster()
        obj.model=m; obj.min_=meta["min"]; obj.max_=meta["max"]; obj.cfg.window=meta["window"]
        return obj

# --- Wrapper ---
def train_lstm_model(df, ticker: str):
    import os
    os.makedirs("data/models", exist_ok=True)

    model=LSTMForecaster(LSTMConfig())
    y=df["log_return"] if "log_return" in df.columns else df["close"]
    model.fit(y.astype(float))

    out=f"data/models/{ticker.replace('.', '_')}_lstm.keras"
    model.save(out)
    print(f"[INFO] ✅ LSTM model saved to {out}")
    return model

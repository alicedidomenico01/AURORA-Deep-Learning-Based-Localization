import pandas as pd

class WeatherScalarLogger:
    def __init__(self):
        self.rows = []

    def log_pairs(
        self,
        split,
        epoch,
        step,
        s_weather_pred,
        s_illum_pred,
        q_cam_pred=None,
        s_weather_gt=None,
        s_illum_gt=None,
        pair_b=None,
        pair_t=None,
        seq_names=None,
        extra_info=None,
    ):
        """
        Log per-pair (P righe per batch camera).
        Tutti i tensori dovrebbero essere su CPU (o convertibili con .tolist()).

        s_weather_pred, s_illum_pred: (P,)
        s_weather_gt, s_illum_gt: (P,) oppure None
        pair_b, pair_t: (P,) oppure None
        seq_names: lista di lunghezza B, dove B è batch di sequenze (indice = pair_b)
        """
        if extra_info is None:
            extra_info = {}

        swp = s_weather_pred.tolist()
        slp = s_illum_pred.tolist()
        P = len(swp)

        if s_weather_gt is not None:
            swg = s_weather_gt.tolist()
        else:
            swg = [None] * P

        if s_illum_gt is not None:
            slg = s_illum_gt.tolist()
        else:
            slg = [None] * P

        if q_cam_pred is not None:
            qcp = q_cam_pred.tolist()
        else:
            qcp = [None] * P

        if pair_b is not None:
            pb = pair_b.tolist()
        else:
            pb = [None] * P

        if pair_t is not None:
            pt = pair_t.tolist()
        else:
            pt = [None] * P

        for i in range(P):
            seq_name = None
            if (seq_names is not None) and (pb[i] is not None):
                try:
                    seq_name = seq_names[int(pb[i])]
                except Exception:
                    seq_name = None

            row = {
                "split": split,
                "epoch": int(epoch),
                "step": int(step),
                "pair_idx": int(i),

                "pair_b": None if pb[i] is None else int(pb[i]),
                "pair_t": None if pt[i] is None else int(pt[i]),
                "seq_name": seq_name,

                "s_weather_pred": float(swp[i]),
                "s_illum_pred": float(slp[i]),

                "q_cam_pred": None if qcp[i] is None else float(qcp[i]),


                "s_weather_gt": None if swg[i] is None else float(swg[i]),
                "s_illum_gt": None if slg[i] is None else float(slg[i]),
            }

            # errori utili per analisi post
            if row["s_weather_gt"] is not None:
                row["err_weather_abs"] = abs(row["s_weather_pred"] - row["s_weather_gt"])
            else:
                row["err_weather_abs"] = None

            if row["s_illum_gt"] is not None:
                row["err_illum_abs"] = abs(row["s_illum_pred"] - row["s_illum_gt"])
            else:
                row["err_illum_abs"] = None

            row.update(extra_info)
            self.rows.append(row)

    def save(self, path="env_scalars_all.csv"):
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")

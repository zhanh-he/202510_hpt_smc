# !/bin/bash
python pytorch/train_score_inf.py model.type="hpt"\
  score_informed.method="direct_output" \
  model.input2=null model.input3=null \
  loss.loss_type="score_inf_custom"

# --------- SCRR ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="scrr" \
  model.input2=onset model.input3=null \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="scrr" \
  model.input2=onset model.input3=frame \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="scrr" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="score_inf_custom"

# --------- Dual Gated ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="dual_gated" \
  model.input2=onset model.input3=null \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="dual_gated" \
  model.input2=onset model.input3=frame \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="dual_gated" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="score_inf_custom"


# --------- Note Editor ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="note_editor" \
  model.input2=onset model.input3=null \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="note_editor" \
  model.input2=onset model.input3=frame \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="note_editor" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="score_inf_custom"

# --------- BiLSTM ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="bilstm" \
  model.input2=onset model.input3=null \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="bilstm" \
  model.input2=onset model.input3=frame \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="bilstm" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="score_inf_custom"

 # --------- Dynest ---------
python pytorch/train_score_inf.py \
  score_informed.method="direct_output" \
  model.input2=null \
  model.input3=null \
  model.type="hppnet" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="scrr" \
  model.input2=onset \
  model.input3=frame \
  model.type="hppnet" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="dual_gated" \
  model.input2=onset \
  model.input3=frame \
  model.type="hppnet" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="note_editor" \
  model.input2=onset \
  model.input3=frame \
  model.type="hppnet" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="bilstm" \
  model.input2=onset \
  model.input3=frame \
  model.type="hppnet" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="direct_output" \
  model.input2=null \
  model.input3=null \
  model.type="dynest" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="scrr" \
  model.input2=onset \
  model.input3=frame \
  model.type="dynest" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="dual_gated" \
  model.input2=onset \
  model.input3=frame \
  model.type="dynest" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="note_editor" \
  model.input2=onset \
  model.input3=frame \
  model.type="dynest" \
  loss.loss_type="score_inf_custom"

python pytorch/train_score_inf.py \
  score_informed.method="bilstm" \
  model.input2=onset \
  model.input3=frame \
  model.type="dynest" \
  loss.loss_type="score_inf_custom"

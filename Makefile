.PHONY: data baseline compare calibrate explain simulate test app final
data:      ; python -m src.data
baseline:  ; python -m src.train baseline
compare:   ; python -m src.train compare
calibrate: ; python -m src.calibrate
explain:   ; python -m src.explain
simulate:  ; python -m src.policy
test:      ; pytest -q
app:       ; streamlit run app/streamlit_app.py
final:     ; python -m src.evaluate

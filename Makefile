.PHONY: basic facial

basic:
	python main.py --task classification --model_name efficientnet_l3 --experiment_name kibz

facial:
	python main.py --task face_recognition --model_name arcface --experiment_name face_recognition
	
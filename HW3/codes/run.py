import os

def run_one_rnn():
    # train
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell rnn")
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell gru")
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell lstm")
    # test
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell rnn")
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell gru")
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 1 --units 300 --decode_strategy random --cell lstm")

def run_two_rnn():  # lstm | gru
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell gru")
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell lstm")
    # test
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell gru")
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell lstm")

def run_strategy():  
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell gru --temperature 0.8")
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy top-p --cell gru --temperature 0.8")
    os.system(
        "python main.py --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy top-p --cell gru --temperature 0.8 --max_probability 0.8")
    # test
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy random --cell gru --temperature 0.8")
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy top-p --cell gru --temperature 0.8")
    os.system(
        "python main.py --test run --num_epochs 100 --batch_size 32 --layers 2 --units 300 --decode_strategy top-p --cell gru --temperature 0.8 --max_probability 0.8")

if __name__ == "__main__":
    from multiprocessing import Process
    p1 = Process(target=run_one_rnn)
    p2 = Process(target=run_two_rnn)
    p3 = Process(target=run_strategy)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


import importlib

def main():
    tests = {
        "DNN": False,
        "CNN": True
    }

    for test in tests:
        if tests[test]:
            module = importlib.import_module(test)
            func = getattr(module, "run")
            func()

    return



if __name__ == "__main__":
    main()
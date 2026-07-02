from mytorch import Tensor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scaler_convert(lst):
    return [ele.data for ele in lst]

def singular(integer_test=True, float_test=True):
    info = np.iinfo(np.intp)
    saved_tensors = []
    saved_arrays = []
    iters = 100
    edge_values = [
            0.0,
            -0.0,
            1e-30,
            1e30,
            np.inf,
            -np.inf,
            np.nan
        ]
    if integer_test:
        print("="*35, "Integer additions","="*35)
        for i in range(iters):
            n1 = np.random.randint(info.min, info.max)
            n2 = np.random.randint(info.min, info.max)
            t1 = Tensor(n1)
            t2 = Tensor(n2)
            try:
                t3 = t1 + t2
                n3 = n1 + n2
                saved_tensors.append(t3)
                saved_arrays.append(n3)
            except TypeError as e:
                return f"Addition failed at iteration {i}: {e}"
            except Exception as e:
                return f"Unexpected error at iteration {i}: {e}"
        
        saved_tensors = scaler_convert(saved_tensors)
        logging.info(f"Success: All {iters} tensor integer additions completed without errors!")
        logging.info(f"validating answers")
        try:
            assert saved_tensors == list(saved_arrays)
        except AssertionError:
            logging.info(f"Error | result mismatch")
        else:
            logging.info(f"Validation Successful, result matched")
        logging.info(f"Max addition for integer additions: {max(saved_arrays)}")
        print(f"-"*100)

    if float_test:
        print("="*35, "Floating point additions","="*35)

        saved_tensors = []
        saved_arrays = []
        for i in range(iters):
            n1 = np.random.uniform(-1, 1)
            n2 = np.random.uniform(-1, 1)
            t1 = Tensor(n1)
            t2 = Tensor(n2)
            try:
                t3 = t1 + t2
                n3 = n1 + n2
                saved_tensors.append(t3)
                saved_arrays.append(n3)
            except Exception as e:
                return f"Addition failed at iteration {i}: {e}"

        logging.info(f"Success: All {iters} tensor floating point additions completed without error!")

        saved_tensors = scaler_convert(saved_tensors)
        logging.info(f"Validating the answers")
        
        equivalent = np.allclose(saved_arrays, saved_tensors)
        if equivalent:
            logging.info(f"Validation successful, results match")
        else:
            logging.info(f"Validation failed, mismatched results")

        logging.info(f"Max addition for floating point operations: {max(saved_tensors)}")
        print(f"-"*100)


def vector(integer_test=True, float_test=True):
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    info = np.iinfo(np.intp)
    add = lambda x, y: x + y
    iters = 100
    def convert_to_list(nlst):
        n = nlst.numel()
        lst = []
        for i in range(n):
            lst.append(nlst.data[i])
        return lst
    if integer_test:
        print("="*35, "starting vector integer addition test", "="*35)
        for i in dims:
            failed = False
            for j in range(iters):
                a1 = [np.random.randint(info.min, info.max) for _ in range(i)]
                a2 = [np.random.randint(info.min, info.max) for _ in range(i)]
                t1 = Tensor(a1)
                t2 = Tensor(a2)
                try:
                    t3 = t1 + t2
                    a3 = [add(a, b) for a, b in zip(a1, a2)]
                except Exception as e:
                    failed = True
                    logging.info(f"Vector Addition error at iteration {j} for values {a1} and {a2}")
                try:
                    assert a3 == t3.data
                except AssertionError as a:
                    failed = True
                    logging.info(f"Equality assertion error for {t1} + {t2}: {a}")
            if not failed:
                logging.info(f"Addition operation on {i} size Tensor successful. \tAnswer successfully asserted!")
        print("="*100)
    
    if float_test:
        print("="*35, "starting floating integer addition test", "="*35)
        for i in dims:
            failed = False
            for j in range(iters):
                a1 = [np.random.uniform(-1, 1) for _ in range(i)]
                a2 = [np.random.uniform(-1, 1) for _ in range(i)]
                t1 = Tensor(a1)
                t2 = Tensor(a2)
                try:
                    t3 = t1 + t2
                    a3 = [add(a,b) for a, b in zip(a1, a2)]
                    assert np.allclose(a3, t3.data)
                except Exception as e:
                    failed = True
                    logging.info(f"Vector Float addition for {i} dimensional Tensor error at iteration {j} for values {t1} and {t2}; error: {a}")
                except AssertionError as ae:
                    failed = True
                    logging.info(f"Floating point addition for {i} dim vector failed. Entries: {t1}+{t2}; error: {ae}")
            if not failed:
                logging.info(f"{i}th dimensional floating point vector addition test successful!")
        print("="*100)




if __name__ == "__main__":
    singular()
    vector()
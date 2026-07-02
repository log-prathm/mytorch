from mytorch import Tensor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

info = np.iinfo(np.intp)

def _element_wise_multiplication(lst_data1, lst_data2):
    if not isinstance(lst_data1, list):
        return lst_data1*lst_data2
    return [_element_wise_multiplication(data1, data2) for data1, data2 in zip(lst_data1, lst_data2)]

def scaler():
    status = True
    for i in range(100):   
        n1 =np.random.randint(info.min, info.max) 
        n2 =np.random.randint(info.min, info.max)

        t1 = Tensor(n1)
        t2 = Tensor(n2)

        try:
            assert n1*n2 == (t1*t2).data
        except AssertionError:
            logging.info(msg=f"Error at scaler multiplication of tensors")
            status = False
        try:
            n1 = np.random.randint(info.min, info.max)
            t3 = t1*n1
            t4 = n1 * t2
            assert t3.data == t1.data*n1
            assert t4.data == n1*t2.data
        except AssertionError as ae:
            status = False
            logging.info(f"Failed result for scaler multiplication of vector with another integer, {ae}")
        except Exception as e:
            status = False
            logging.info(f"Failed scaler multiplication of vector with another integer, {e}")
    if status:
        logging.info(f"Successfully conducted scaler addition test!")
    else:
        logging.info(f"Error in conducting scaler addition!")


def vector():
    dims = [2, 8, 16, 64, 512]
    iters = 100
    status = False
    for i in dims:
        for j in range(iters):
            s1=[np.random.randint(info.min, info.max) for _ in range(i)]
            s2=[np.random.randint(info.min, info.max) for _ in range(i)]
            t1 = Tensor(s1)
            t2 = Tensor(s2)

            try:
                assert _element_wise_multiplication(s1, s2) == (t1*t2).data
            except AssertionError as e:
                status = True
                logging.info(f"Wrong result in vector multiplication test at dim {i} and tensors {t1} and {t2}, e")
            except Exception as e:
                status = True
                logging.info(f"Error occured while conducting vector multiplication test at dim {i} vector, {e}")
    if not status:
        logging.info(f"Vector multiplication test successfully conducted!")
            

    return t1*t2
if __name__ == "__main__":
    scaler()

    # s1=[np.random.randint(info.min, info.max) for _ in range(2)]
    # s2=[np.random.randint(info.min, info.max) for _ in range(2)]
    # t1 = Tensor(s1)
    # t2 = Tensor(s2)
    

    # s3 = _element_wise_multiplication(s1, s2)
    # t3 = t1*t2
    # flat = s3 == t3.data
    # print(flat)
    # print(s3)
    # print(t3)

    # a = Tensor([1,2])
    # b = Tensor([[1,2]])
    # print(a+b)
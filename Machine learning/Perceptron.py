import numpy as np

class Perceptron(object):
    """퍼셉트론 분류기
    
    매개변수
    --------
    eta : float
        학습률 (0.0과 1.0 사이)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드
        일단 학습이니까, 똑같은 학습 결과를 재현할 수 있도록
    속성
    --------
    w_ : 1d-array
        학습된 가중치
    errors_ : list
        에포크마다 누적된 분류 오류
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        """
        self는  지금까지 봤던 클래스 안에 있던 self에 대해 찾아보니  "인스턴스 자신", "그 시점의 자신", "메소드의 임의의 인수"등 다양하게 부르고 있었다.
        그러나 클래스의 구성을 취득할 때에 정형의 구문으로써 기억해두면 괜찮은 것 같다.
        출처: https://engineer-mole.tistory.com/190 [매일 꾸준히, 더 깊이:티스토리]
        
        클래스
        """
    def fit(self, X, y):
        """
        훈련데이터 학습
        
        매개변수
        --------
        X:{araay-like}, shape=[n_samples, n_features] (row, column)
          n_sample개의 샘플과 n_featrues개의 특성으로 이루어진 훈련 데이터
        y:array-like, shape = {n_samples}
          타깃 값
          
        반환 값
        -------
        self:object
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        """0에서 1사이의 난수를 발생, scale은 0.01, 갯수는 특성 갯수보다 더 많게, 차원 + 1, 절편 구현"""
        self.errors_ = [] 
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y): 
                """tuple 만들기"""
                update = self.eta * (target - self.predict(xi)) 
                """y_hat을 만드는 것"""
                self.w_[1:] += update * xi
                self.w_[0] += update 
                
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        #입력 계산
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        #단위 계산 함수를 사용하여 클래스 레이블을 반환합니다.
        return np.where(self.net_input(X) >= 0.0, 1, -1)
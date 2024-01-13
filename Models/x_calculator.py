import math

PRINCIPAL = 10_00_000
Z_FACTOR = 5
K_VALUE = 0.8
ERI = 0.05
EFI = 0
THRESHOLD = 0.75


def calculate_x(data):
    beta = 1
    k = K_VALUE * data["emissions"] + 1
    E_r_i = ERI * PRINCIPAL
    E_f_i = EFI
    I_in = len(data[data["industry"] == data["industry"]])
    D_sp = {"Low": 0, "Medium": 1, "High": 2}.get(data["disaster_risk"], 0)

    gamma = data["importance"] if data["importance"] != 0 else 0.001
    denominator = max(k * gamma * math.log(abs(k * gamma)), 1)

    numerator = PRINCIPAL * Z_FACTOR
    InvestmentVsCapital = math.log(I_in / PRINCIPAL) if I_in > PRINCIPAL else 0

    X = (
            beta / PRINCIPAL * ((numerator / denominator) - E_r_i * k)
            - E_f_i
            + InvestmentVsCapital
            - (D_sp * 1000)
    )
    return X

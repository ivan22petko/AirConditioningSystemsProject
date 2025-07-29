import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

# constants
c = 1e3         # air specific heat J/kg K
l = 2496e3      # latent heat J/kg

def ModelECS(m_fresh, α, θS, θIsp, φIsp, θ_turb, φ_turb, θ_ext, Qsa, Qla):
    """
    INPUTS:
        m_fresh   fresh air mass flow from ACM kg/s
        α   mixing ratio
        θS   supply air temperature °C
        θIsp  indoor air temperature setpoint °C
        φIsp indoor relative humidity setpoint -
        θ_turb   air temperature after turbine °C
        φ_turb   relative humidity after turbine -
        θ_ext   external air static temperature °C
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W

    OUTPUTS:
        x     vector 12 elements:
            θ0, w0, θ1, w1, t2, w2, t3, w3, QsHC, QlVH, QsTZ, QlTZ

    System:
        MX:     Mixing Box
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Buildings
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions


         <-3--|<-----------------------|
              |                        |
    -turbine->MX--0->HC--1->VH--2->TZ--3-->
                     /       /     ||  |
                     |       |     BL  |
                     |       |         |
                     |       |<----Kw--|-w3
                     |<------------Kt--|-t3
    """

    m = m_fresh / α

    QsZ = -m*c*(θS - θIsp)

    UA = (QsZ - Qsa)/(θ_ext-θIsp)

    Kt, Kw = 1e10, 1e10             # controller gain
    w_turb = psy.w(θ_turb, φ_turb)            # hum. out
    wIsp = psy.w(θIsp, φIsp)      # hum. in set point

    # Model
    A = np.zeros((12, 12))          # coefficents of unknowns
    b = np.zeros(12)                # vector of inputs
    # MX mixing box
    A[0, 0], A[0, 6], b[0] = m * c, -(1 - α) * m * c, α * m * c * θ_turb
    A[1, 1], A[1, 7], b[1] = m * l, -(1 - α) * m * l, α * m * l * w_turb
    # HC heating coil
    A[2, 0], A[2, 2], A[2, 8], b[2] = m * c, -m * c, 1, 0
    A[3, 1], A[3, 3], b[3] = m * l, -m * l, 0
    # VH vapor humidifier
    A[4, 2], A[4, 4], b[4] = m * c, -m * c, 0
    A[5, 3], A[5, 5], A[5, 9], b[5] = m * l, -m * l, 1, 0
    # TZ thermal zone
    A[6, 4], A[6, 6], A[6, 10], b[6] = m * c, -m * c, 1, 0
    A[7, 5], A[7, 7], A[7, 11], b[7] = m * l, -m * l, 1, 0
    # BL building
    A[8, 6], A[8, 10], b[8] = UA, 1, UA * θ_ext + Qsa
    A[9, 7], A[9, 11], b[9] = 0, 1, Qla
    # Kt indoor temperature controller
    A[10, 6], A[10, 8], b[10] = Kt, 1, Kt * θIsp
    # Kw indoor humidity controller
    A[11, 7], A[11, 9], b[11] = Kw, 1, Kw * wIsp

    # Solution
    x = np.linalg.solve(A, b)
    
    A = np.array([[-1, 1, 0, 0, -1],        # MX
                 [0, -1, 1, 0, 0],          # HC
                 [0, 0, -1, 1, 0],          # VH
                 [0, 0, 0, -1, 1]])         # TZ
    t = np.append(θ_turb, x[0:8:2])

    w = np.append(w_turb, x[1:8:2])
    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000 * pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['θ [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        'θ [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[8:], index=['QsHC', 'QlVH', 'QsTZ', 'QlTZ'])
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T / 1000, 'kW')

    print("UA:", round(UA, 0))

    return x
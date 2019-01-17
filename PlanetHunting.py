import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

z0,time,flux=np.loadtxt("DATA_FILE_PATH",unpack=True,skiprows=3) # Loads time & flux data from file.

if __name__ == '__main__':
    plt.plot(time,flux,"b.")                   # Plots graph of raw data.
    plt.xlabel("Time (Julian Days)")
    plt.ylabel("Relative Flux")
    plt.show()

transValAvgs = []
transVals = []
tolerance = float(input("Tolerance: "))     # User chooses tolerance for deciding data points being considered as "during transit".

for i in range(0, len(flux)):
    if flux[i] > -tolerance:
        transVals = []
    if flux[i] <= -tolerance:
        transVals.append(time[i])
        if flux[i + 1] > -tolerance:
            transValAvgs.append(np.mean(transVals))

periodSet = []                                          # Calculates period as an average of the time difference between transit midpoints.
for i in range(0, (len(transValAvgs) - 1)):
	periodSet.append(transValAvgs[i + 1] - transValAvgs[i])

period = np.mean(periodSet)

offset = transValAvgs[0]

time2 = []
phase = []

for i in range(0, len(time)):                           # Phase-folds the time series, overlapping each of the transits.
    time2.append((time[i] - offset) % period)
    if time2[i] > (0.5 * period):
        phase.append(time2[i] - period)
    else:
        phase.append(time2[i])

phase, flux = zip(*sorted(zip(phase, flux)))

if __name__ == '__main__':
    plt.plot(phase, flux, "b.")
    plt.xlabel("Time (Julian Days)")
    plt.ylabel("Relative Flux")
    plt.show()

tIngS = []
tIngE = []
maxD = []
fIngressStart = 0

tIngressStartGuess = float(input("Guess for ingress start: "))
tIngressEndGuess = float(input("Guess for ingress end: "))
maxDepthGuess = float(input("Guess for maximum depth: "))

for i in range(0, 11):                                      # Generates values around the initial guesses.
    if i == 0:
        tIngS.append(tIngressStartGuess)
        tIngE.append(tIngressEndGuess)
        maxD.append(maxDepthGuess)
    else:
        tIngS.append(tIngressStartGuess + (i / 10000))
        tIngS.append(tIngressStartGuess - (i / 10000))

        tIngE.append(tIngressEndGuess + (i / 10000))
        tIngE.append(tIngressEndGuess - (i / 10000))

        maxD.append(maxDepthGuess + (i / 10000))
        maxD.append(maxDepthGuess - (i / 10000))

sorted(tIngS)
sorted(tIngE)
sorted(maxD)

fluxSet = []
varPoint = 0

while phase[varPoint] < tIngressStartGuess:
    fluxSet.append(flux[varPoint])
    varPoint += 1

fluxVar = np.var(fluxSet)


def ChiSqrCalc(sIndex, fIndex, modelListProcess, indexListProcess, chiSqrListProcess):          # Function which generates a model for all possible combinations of the values generated above.

    for i in range(sIndex, fIndex):
        modelList = []
        indexList = []
        chiSqrList = []
        tIngressS = tIngS[i]

        for j in range(0, len(tIngE)):
            tIngressE = tIngE[j]
            if tIngressE < tIngressS:
                continue

            for k in range(0, len(maxD)):
                maxDep = maxD[k]
                gradIngress = maxDep / (tIngressE - tIngressS)
                model = []
                chiSqrN = 0

                for l in range(0, len(flux)):                                                   # A simple trapezoid model is generated to fit the data.
                    if phase[l] < tIngressS:
                        model.append(0)

                    elif tIngressS <= phase[l] <= tIngressE:
                        model.append((gradIngress * phase[l]) + (fIngressStart - gradIngress * tIngressS))

                    elif tIngressE <= phase[l] <= -tIngressE:
                        model.append(maxDep)

                    elif -tIngressE <= phase[l] <= -tIngressS:
                        model.append((-gradIngress * phase[l]) + (fIngressStart - gradIngress * tIngressS))

                    else:
                        model.append(0)

                    chiSqrN += ((flux[l] - model[l]) ** 2) / fluxVar

                modelList.append(model)
                indexList.append([i, j, k])
                chiSqrList.append(chiSqrN / len(flux))                                      #Chi-squard test is carried out for each model.

        chiMin = np.argmin(chiSqrList)
        modelListProcess.append(modelList[chiMin])
        indexListProcess.append(indexList[chiMin])
        chiSqrListProcess.append(chiSqrList[chiMin])
        

if __name__ == '__main__':                                                                  # The above function is carried out across three processes, to reduce time consumption.
    
    manager = Manager()
    mListP1 = manager.list()
    iListP1 = manager.list()
    cListP1 = manager.list()
    mListP2 = manager.list()
    iListP2 = manager.list()
    cListP2 = manager.list()
    mListP3 = manager.list()
    iListP3 = manager.list()
    cListP3 = manager.list()

    p1 = Process(target=ChiSqrCalc, args=(0, 7, mListP1, iListP1, cListP1))
    p2 = Process(target=ChiSqrCalc, args=(7, 14, mListP2, iListP2, cListP2))
    p3 = Process(target=ChiSqrCalc, args=(14, 21, mListP3, iListP3, cListP3))

    processes = [p1, p2, p3]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modelArr = mListP1
    index = iListP1
    chiSqr = cListP1

    chiSqrMinIndex = np.argmin(cListP1)
    chiSqrMin = cListP1[chiSqrMinIndex]

    if cListP2[np.argmin(cListP2)] < chiSqrMin:
        modelArr = mListP2
        index = iListP2
        chiSqr = cListP2

        chiSqrMinIndex = np.argmin(cListP2)
        chiSqrMin = cListP2[chiSqrMinIndex]

    if cListP3[np.argmin(cListP3)] < chiSqrMin:
        modelArr = mListP3
        index = iListP3
        chiSqr = cListP3

        chiSqrMinIndex = np.argmin(cListP3)
        chiSqrMin = cListP3[chiSqrMinIndex]

    modelFin = modelArr[chiSqrMinIndex]                                                         # The final combination of values are chosen as those that produce the model with the minimum Chi-squared value.
    tIngressStart = tIngS[index[chiSqrMinIndex][0]]
    tIngressEnd = tIngE[index[chiSqrMinIndex][1]]
    maxDepth = maxD[index[chiSqrMinIndex][2]]

    rad, log10g = np.loadtxt("DATA_FILE_PATH", unpack=True)                                     # Star radius and surface gravity are loaded from a second file.

    rSun = 6.95700 * 10 ** 8                                                                    # Stellar and planetary parameters are calculated, and output to the terminal.
    rStar = rad * rSun

    G = 6.67408 * 10 ** (-11)
    g = (10 ** log10g) / 100
    starMass = (g * rStar ** 2) / G
    solMass = 1.98847542 * 10 ** 30

    rPlan = rStar * np.sqrt(-maxDepth)
    rPlanEarth = rPlan / (6.3781 * 10 ** 6)

    periodSec = period * 24 * 60 ** 2
    semMajAxis = np.cbrt((g * (rStar * periodSec) ** 2) / (4 * np.pi ** 2))
    semMajAxisAU = semMajAxis / (1.496 * 10 ** 11)

    tDur = 2 * (-tIngressStart) * 24 * 60 ** 2
    combinedRad = rStar + rPlan
    sinTerm = semMajAxis * np.sin(tDur * np.pi / periodSec)
    bR = np.sqrt(combinedRad ** 2 - sinTerm ** 2)
    impParam = bR / rStar

    inclin = np.arccos(bR / semMajAxis)
    inclinDeg = inclin * 180 / np.pi

    print("Chi squared =\t", chiSqrMin, "\n")
    print("Using the stellar parameters:")
    print("Stellar radius=\t", rad, "Solar radii")
    print("log10(g)=\t", log10g, "\n")
    print("Stellar Mass=\t", starMass / solMass, "Solar masses\n")
    print("Planetary parameters:")
    print("Planetary radius=\t", rPlanEarth, "Earth radii")
    print("Semi-major axis=\t", semMajAxisAU, "AU")
    print("Impact parameter=\t", impParam)
    print("Orbital inclination (rad)=\t", inclin)
    print("Orbital inclination (deg)=\t", inclinDeg)
    
    plt.plot(phase, flux, "b.")                                                                 # Final model is plotted, in conjunction with the phase-folded time series.
    plt.plot(phase, modelFin, "r")
    plt.xlabel("Time (Julian Days)")
    plt.ylabel("Relative Flux")
    plt.show()

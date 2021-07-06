import ctypes as ct
import os
import numpy as np
from glob import glob

c_int = ct.c_int
c_double = ct.c_double
c_void_p = ct.c_void_p
c_char_p = ct.c_char_p
c_double_pointer = ct.POINTER(c_double)
c_int_pointer = ct.POINTER(c_int)
libname = glob(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastrat.*.so")
)[0]


class snake:
    def __init__(self):
        self.lib = ct.cdll.LoadLibrary(libname)
        self._init_functions()

    def _init_functions(self):
        self.l_square = self.lib.square
        self.l_square.restype = c_double_pointer
        self.l_square.argstypes = [c_double_pointer, c_int]

        self._freeSquare = self.lib.freeSquare
        self._freeSquare.restype = c_void_p
        self._freeSquare.argstypes = []

        self._openFile = self.lib.openFile
        self._openFile.restype = c_void_p
        self._openFile.argstypes = [c_char_p]

        self._getEntries = self.lib.getEntries
        self._getEntries.restype = c_int
        self._getEntries.argstypes = []

        self._getBoundaryRadius = self.lib.getBoundaryRadius
        self._getBoundaryRadius.restype = c_double
        self._getBoundaryRadius.argstypes = []

        self._getBoundaryHalfz = self.lib.getBoundaryHalfz
        self._getBoundaryHalfz.restype = c_double
        self._getBoundaryHalfz.argstypes = []

        self._getEvent = self.lib.getEvent
        self._getEvent.restype = c_void_p
        self._getEvent.argstypes = [c_int]

        self._getNHit = self.lib.getNHit
        self._getNHit.restype = c_int
        self._getNHit.argstypes = []
        self._getPMTCount = self.lib.getPMTCount
        self._getPMTCount.restype = c_int
        self._getPMTCount.argstypes = []
        self._getPMTX = self.lib.getPMTX
        self._getPMTX.restype = c_double_pointer
        self._getPMTX.argstypes = []
        self._getPMTY = self.lib.getPMTY
        self._getPMTY.restype = c_double_pointer
        self._getPMTY.argstypes = []
        self._getPMTZ = self.lib.getPMTZ
        self._getPMTZ.restype = c_double_pointer
        self._getPMTZ.argstypes = []
        self._getPMTNoise = self.lib.getPMTNoise
        self._getPMTNoise.restype = c_int_pointer
        self._getPMTNoise.argstypes = []
        self._getPMTType = self.lib.getPMTType
        self._getPMTType.restype = c_int_pointer
        self._getPMTType.argstypes = []
        self._getCharge = self.lib.getCharge
        self._getCharge.restype = c_double_pointer
        self._getCharge.argstypes = []
        self._getTime = self.lib.getTime
        self._getTime.restype = c_double_pointer
        self._getTime.argstypes = []

        self._getTracking = self.lib.getTracking
        self._getTracking.restype = c_void_p
        self._getTracking.argstypes = []
        self._getTrackCount = self.lib.getTrackCount
        self._getTrackCount.restype = c_int
        self._getTrackCount.argstypes = []
        self._getTrackX = self.lib.getTrackX
        self._getTrackX.restype = c_double_pointer
        self._getTrackX.argstypes = []
        self._getTrackY = self.lib.getTrackY
        self._getTrackY.restype = c_double_pointer
        self._getTrackY.argstypes = []
        self._getTrackZ = self.lib.getTrackZ
        self._getTrackZ.restype = c_double_pointer
        self._getTrackZ.argstypes = []
        self._getTrackNames = self.lib.getTrackNames
        self._getTrackNames.restype = c_int_pointer
        self._getTrackNames.argstypes = []
        
        self._getMCTruth = self.lib.getMCTruth
        self._getMCTruth.restype = c_void_p
        self._getMCTruth.argstypes = []
        self._getMCX = self.lib.getMCX
        self._getMCX.restype = c_double_pointer
        self._getMCX.argstypes = []
        self._getMCY = self.lib.getMCY
        self._getMCY.restype = c_double_pointer
        self._getMCY.argstypes = []
        self._getMCZ = self.lib.getMCZ
        self._getMCZ.restype = c_double_pointer
        self._getMCZ.argstypes = []
        self._getMCU = self.lib.getMCU
        self._getMCU.restype = c_double_pointer
        self._getMCU.argstypes = []
        self._getMCV = self.lib.getMCV
        self._getMCV.restype = c_double_pointer
        self._getMCV.argstypes = []
        self._getMCW = self.lib.getMCW
        self._getMCW.restype = c_double_pointer
        self._getMCW.argstypes = []
        self._getMCKE = self.lib.getMCKE
        self._getMCKE.restype = c_double_pointer
        self._getMCKE.argstypes = []
        self._getMCT = self.lib.getMCT
        self._getMCT.restype = c_double_pointer
        self._getMCT.argstypes = []
        self._getPDG = self.lib.getPDG
        self._getPDG.restype = c_int_pointer
        self._getPDG.argstypes = []
        self._getMCCount = self.lib.getMCCount
        self._getMCCount.restype = c_int
        self._getMCCount.argstypes = []
        
        self._maskPMT = self.lib.maskPMT
        self._maskPMT.restype = c_void_p
        self._getEvent.argstypes = [c_int]
        
        self._writeFile = self.lib.writeFile
        self._writeFile.restype = c_void_p
        self._writeFile.argstypes = [c_char_p]
        
    def openFile(self, fname):
        print(fname)
        self._openFile(fname.encode())

    def getEvent(self, idx):
        self._getEvent(idx)

    def getEntries(self):
        return self._getEntries()

    def getBoundaryRadius(self):
        return self._getBoundaryRadius()

    def getBoundaryHalfz(self):
        return self._getBoundaryHalfz()

    def getXYZ(self):
        pmtc = self._getPMTCount()
        x_ret = self._getPMTX()
        x_arr = np.fromiter(x_ret, dtype=np.float64, count=pmtc)
        y_ret = self._getPMTY()
        y_arr = np.fromiter(y_ret, dtype=np.float64, count=pmtc)
        z_ret = self._getPMTZ()
        z_arr = np.fromiter(z_ret, dtype=np.float64, count=pmtc)
        return x_arr, y_arr, z_arr

    def getID(self):
        pmtc = self._getPMTCount()
        type_ret = self._getPMTType()
        type_arr = np.fromiter(type_ret, dtype=np.int64, count=pmtc)
        
        return type_arr
        
    def getHitInfo(self):
        pmtc = self._getPMTCount()
        c_ret = self._getCharge()
        c_arr = np.fromiter(c_ret, dtype=np.float64, count=pmtc)
        t_ret = self._getTime()
        t_arr = np.fromiter(t_ret, dtype=np.float64, count=pmtc)
        noise_ret = self._getPMTNoise()
        noise_arr = np.fromiter(noise_ret, dtype=np.int64, count=pmtc)
        
        return c_arr, t_arr, noise_arr

    def square(self, arr):
        arr = np.array(arr, dtype=np.float64)
        asend = arr.ctypes.data_as(c_double_pointer)
        areturn = self.l_square(asend, len(arr))
        newarr = np.fromiter(areturn, dtype=np.float64, count=len(arr))
        self._freeSquare()
        return newarr

    ## Tracking information
    def getTracking(self):
        self._getTracking()

    def getTrackSteps(self):
        tsteps = self._getTrackCount()
        x_ret = self._getTrackX()
        x_arr = np.fromiter(x_ret, dtype=np.float64, count=tsteps)
        y_ret = self._getTrackY()
        y_arr = np.fromiter(y_ret, dtype=np.float64, count=tsteps)
        z_ret = self._getTrackZ()
        z_arr = np.fromiter(z_ret, dtype=np.float64, count=tsteps)
        n_ret = self._getTrackNames()
        n_arr = np.fromiter(n_ret, dtype=np.int64, count=tsteps)
        return x_arr, y_arr, z_arr, n_arr

    ## Incident Particle Information
    def getMCTruth(self):
        self._getMCTruth()
    
    def getMCVal(self):
        p = self._getMCCount()
        x_arr = np.fromiter(self._getMCX(), dtype=np.float64, count=p)
        y_arr = np.fromiter(self._getMCY(), dtype=np.float64, count=p)
        z_arr = np.fromiter(self._getMCZ(), dtype=np.float64, count=p)
        u_arr = np.fromiter(self._getMCU(), dtype=np.float64, count=p)
        v_arr = np.fromiter(self._getMCV(), dtype=np.float64, count=p)
        w_arr = np.fromiter(self._getMCW(), dtype=np.float64, count=p)
        ke_arr = np.fromiter(self._getMCKE(), dtype=np.float64, count=p)
        t_arr = np.fromiter(self._getMCT(), dtype=np.float64, count=p)
        pdg_arr = np.fromiter(self._getPDG(), dtype=np.float64, count=p)
        
        return x_arr, y_arr, z_arr, u_arr, v_arr, w_arr, t_arr, ke_arr, pdg_arr

    def maskPMT(self,pmtid):
        self._maskPMT(pmtid)
        
        
    def writeFile(self, fname):
        print(fname)
        self._writeFile(fname.encode())

if __name__ == "__main__":
    a = np.random.rand(10)
    cobra = snake()
    cobra.square(a)
    cobra.openFile("cobalt_watchman.root")
    cobra.getEvent(0)

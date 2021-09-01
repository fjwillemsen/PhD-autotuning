

from kernel_tuner.observers import BenchmarkObserver


#get number of registers
class RegisterObserver(BenchmarkObserver):
    def get_results(self):
        return {"num_regs": self.dev.func.num_regs}
reg_observer = RegisterObserver()


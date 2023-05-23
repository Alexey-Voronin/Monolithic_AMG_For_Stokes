import time

timings = None


def timings_data_init():
    global timings_data
    timings_data = {}

def timeit(arg):
    def inner_decorator(f):
        def timeit_wrapper(*args, **kwargs):
            if 'timings_data' not in globals():
                timings_data_init()

            fxn_name = f.__name__
            key = f"{arg}{fxn_name}"

            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            total_time = time.perf_counter() - start_time


            if fxn_name == 'relax_inplace':
                lvl = result
                new_name = 'solution:solver:rlx'

                if new_name + '(c)' in timings_data:
                    timings_data[new_name + '(t)'][lvl] += total_time
                    timings_data[new_name + '(c)'][lvl] += 1
                else:
                    timings_data[new_name + '(t)'] = [0.0] * 10
                    timings_data[new_name + '(c)'] = [0.0] * 10
                    timings_data[new_name + '(t)'][lvl] = total_time
                    timings_data[new_name + '(c)'][lvl] = 1
            elif key in ['solution:solver:wrapper:mv', \
                         'solution:solver:dg_to_cg:mv', \
                         'solution:solver:bd_mg:rlx', \
                         'solution:solver:coarse:coarse_solve', \
                         'setup:solver:vanka:_patch_solver', \
                         'setup:solver:vanka:_get_patch'
                         ]:
                if key in timings_data:
                    timings_data[key] += total_time
                else:
                    timings_data[key] = total_time
            elif key in timings_data:
                timings_data[key].append(total_time)
            else:
                timings_data[key] = [total_time]
            return result

        return timeit_wrapper

    return inner_decorator

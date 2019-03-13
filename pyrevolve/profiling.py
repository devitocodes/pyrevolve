from timeit import default_timer


class Timer(object):
    def __init__(self, profiler, section, action):
        self.timer = default_timer
        self.profiler = profiler
        self.section = section
        self.action = action

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.profiler.increment(self.section, self.action, self.elapsed)


class Profiler(object):
    def __init__(self):
        self.timings = {}
        self.counts = {}

    def get_timer(self, section, action):
        return Timer(self, section, action)

    def increment(self, section, action, elapsed):
        # Warning: Not thread safe
        section_timings = self.timings.get(section, {})
        section_timings[action] = section_timings.get(action, 0) + elapsed
        self.timings[section] = section_timings

        section_counts = self.counts.get(section, {})
        section_counts[action] = section_counts.get(action, 0) + 1
        self.counts[section] = section_counts

    def summary(self):
        summary = '****************'
        for section, section_timings in self.timings.items():
            summary += '\nIn section %s:' % section
            for action, action_time in section_timings.items():
                summary += '\n\tAction %s: %f (%d)' \
                           % (action, action_time,
                              self.counts[section][action])
        summary += '\n****************'
        return summary

    def get_dict(self):
        results = {}
        for s_n, s_dict in self.timings.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_timing' % (s_n, a_n)] = a_time

        for s_n, s_dict in self.counts.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_counts' % (s_n, a_n)] = a_time

        return results

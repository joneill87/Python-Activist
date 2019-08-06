import collections

# Create a struct to hold our Batch Summary object. This simplifies writing to CSV
BatchSummary = collections.namedtuple('BatchSummary', ['batch', 'total', 'isLabelled', 'correct',
                                                       'incorrect', 'truePositive', 'trueNegative',
                                                       'falsePositive', 'falseNegative'])


# Create a struct to hold our Batch Evaluation object. This is useful as documentation for the StoppingCriterion
BatchEvaluation = collections.namedtuple('BatchEvaluation', ['batchNumber', 'id', 'isLabelled', 'confidence', 'prediction', 'label'])

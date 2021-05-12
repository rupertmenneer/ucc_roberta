def make_sensible(r):
    sensible = []
    # for every result
    for i in range(len(r)):
        # create dictionary which is correct format
        # return true/false value depending on classification per label
        sensible.append(
            {'classifier/antagonise': bool(r[i][0]),
             'classifier/condescending': bool(r[i][1]),
             'classifier/dismissive': bool(r[i][2]),
             'classifier/generalisation': bool(r[i][3]),
             'classifier/generalisation_unfair': bool(r[i][4]),
             'classifier/unhealthy': bool(r[i][5]),
             'classifier/hostile': bool(r[i][6]),
             'classifier/sarcastic': bool(r[i][7]),
             },
        )
    return sensible
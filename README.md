<h1>HorsePythons</h1>
<h2>A python environment for predicting horse races</h2>
<strong>***All dates are in YYYYMMDD format***</strong>
<h3>Crawler and options</h3>
<ul>
    <li><code>python crawler.py</code> to grab all Remington Park data</li>
    <li><code>-b BASE_DATE</code> to choose base date from which to take list of race days</li>
    <li><code>-s START_DATE</code> to set a starting bound for the daterange</li>
    <li><code>-e END_DATE</code> to set an ending bound for the daterange</li>
    <li><code>-o OUTFILE</code> to write to a specific outfile</li>
</ul>

<h3>Machine Learning and options</h3>
<ul>
    <li><code>python ai.py -t TRAINING_FILE</code> to train using the data in the training file</li>
    <li><code>-i ITERATIONS</code> to set number of training iterations on data
    <li><code>-q quiet</code> to only print the final output table</li>
    <li><code>-v verbose</code> to print extra messages at runtime</li>
</ul>
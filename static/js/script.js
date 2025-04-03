document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    const resultText = resultDiv.querySelector('.result-text');

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.classList.remove('success', 'error', 'show');
        if (data.error) {
            resultText.textContent = `Oops! Something went wrong: ${data.error}`;
            resultDiv.classList.add('error');
        } else {
            resultText.textContent = `Your food will arrive in approximately ${data.prediction} minutes! Enjoy!`;
            resultDiv.classList.add('success');
        }
        // Trigger animation
        setTimeout(() => resultDiv.classList.add('show'), 10);
    })
    .catch(error => {
        resultDiv.classList.remove('success', 'error', 'show');
        resultText.textContent = `Uh-oh! Couldn’t connect to the server. Please try again.`;
        resultDiv.classList.add('error');
        setTimeout(() => resultDiv.classList.add('show'), 10);
    });
});
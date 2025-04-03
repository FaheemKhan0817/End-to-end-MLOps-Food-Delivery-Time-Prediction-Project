document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.classList.remove('success', 'error');
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
            resultDiv.classList.add('error');
        } else {
            resultDiv.textContent = `Predicted Delivery Time: ${data.prediction} minutes`;
            resultDiv.classList.add('success');
        }
    })
    .catch(error => {
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.classList.add('error');
    });
});
async function predict() {

    const fileInput = document.getElementById("imageInput");

    if(fileInput.files.length === 0){
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try{

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        document.getElementById("result").innerHTML = `
            <h3>${data.predicted_class}</h3>
            <p>Confidence: ${data.confidence * 100}%</p>
        `;

    }catch(error){
        console.error(error);
    }
}
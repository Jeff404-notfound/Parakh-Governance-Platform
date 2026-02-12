fetch("/stats").then(r=>r.json()).then(data=>{
    new Chart(document.getElementById("stateChart"),{
        type:"bar",
        data:{
            labels:Object.keys(data.state),
            datasets:[{data:Object.values(data.state)}]
        }
    });
});

fetch("/problems").then(r=>r.json()).then(data=>{
    let p = document.getElementById("problems");
    data.under_served.forEach(d=>{
        p.innerHTML += "<li>Under-served district: "+d+"</li>";
    });
    data.fraud_centers.forEach(c=>{
        p.innerHTML += "<li>Fraud risk: Centre "+c+"</li>";
    });
});

fetch("/policy").then(r=>r.json()).then(data=>{
    let p = document.getElementById("policy");
    data.forEach(r=>{
        p.innerHTML += "<li>"+r+"</li>";
    });
});

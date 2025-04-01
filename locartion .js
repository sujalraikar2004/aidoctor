function getLocation() {
    navigator.geolocation.getCurrentPosition(
        pos => {
            const coords = `${pos.coords.latitude},${pos.coords.longitude}`;
            document.getElementById("coords").value = coords;
        },
        err => console.error("Geolocation error:", err)
    );
}
window.addEventListener('load', getLocation);
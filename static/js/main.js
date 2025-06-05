// Main JavaScript for SEO Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Handle analyze form submission with AJAX
    const analyzeForm = document.getElementById('analyzeForm');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading spinner
            const spinner = document.querySelector('.loading-spinner');
            if (spinner) {
                spinner.style.display = 'block';
            }
            
            // Get form data
            const formData = new FormData(analyzeForm);
            
            // Send AJAX request
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    // Handle error
                    alert('Error: ' + data.error);
                    if (spinner) {
                        spinner.style.display = 'none';
                    }
                } else if (data.success && data.session_id) {
                    // Redirect to results page
                    window.location.href = '/results/' + data.session_id;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
                if (spinner) {
                    spinner.style.display = 'none';
                }
            });
        });
    }
    
    // Show loading spinner for other forms (batch and compare)
    const otherForms = document.querySelectorAll('form:not(#analyzeForm)');
    otherForms.forEach(form => {
        form.addEventListener('submit', function() {
            const spinner = document.querySelector('.loading-spinner');
            if (spinner) {
                spinner.style.display = 'block';
            }
        });
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize DataTables if present
    if (typeof $.fn.DataTable !== 'undefined' && document.getElementById('resultsTable')) {
        $('#resultsTable').DataTable({
            responsive: true,
            order: [[3, 'desc']], // Order by SEO score column descending
            dom: 'Bfrtip',
            buttons: [
                'copy', 'csv', 'excel', 'pdf'
            ]
        });
    }

    // CSV Export functionality
    const exportBtn = document.getElementById('exportCSV');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            exportTableToCSV('seo_analysis_results.csv');
        });
    }

    // URL comparison text area counter
    const urlTextarea = document.getElementById('urls');
    const urlCounter = document.getElementById('urlCounter');
    if (urlTextarea && urlCounter) {
        urlTextarea.addEventListener('input', function() {
            const urls = this.value.split('\n').filter(url => url.trim() !== '');
            urlCounter.textContent = urls.length;
            
            // Warn if too many URLs
            if (urls.length > 5) {
                urlCounter.classList.add('text-danger');
            } else {
                urlCounter.classList.remove('text-danger');
            }
        });
    }
});

// Function to export table data to CSV
function exportTableToCSV(filename) {
    const table = document.querySelector('table');
    if (!table) return;
    
    let csv = [];
    const rows = table.querySelectorAll('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const row = [], cols = rows[i].querySelectorAll('td, th');
        
        for (let j = 0; j < cols.length; j++) {
            // Replace any commas in the cell text with spaces to avoid CSV formatting issues
            let text = cols[j].innerText.replace(/,/g, ' ');
            // Remove any quotes to avoid CSV formatting issues
            text = text.replace(/"/g, '');
            row.push('"' + text + '"');
        }
        
        csv.push(row.join(','));
    }
    
    // Download CSV file
    downloadCSV(csv.join('\n'), filename);
}

function downloadCSV(csv, filename) {
    const csvFile = new Blob([csv], {type: "text/csv"});
    const downloadLink = document.createElement("a");
    
    // Create a download link
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = "none";
    
    // Add the link to the DOM and trigger the download
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

// Function to download CSV template for batch analysis
function downloadCSVTemplate() {
    const csvContent = "url,query\nhttps://example.com,seo analyzer\nhttps://example.org,web crawler";
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'batch_analysis_template.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Function to create radar charts for comparison
function createRadarChart(elementId, labels, datasets) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });
}

/**
 * PDF Export Module — Reusable PDF generation for any page.
 *
 * Lazy-loads jsPDF + html2canvas on first use (~500KB total).
 * Supports Chart.js canvas capture and Plotly.toImage() conversion.
 *
 * Usage:
 *   import { exportPageToPDF } from './modules/pdf-export.js';
 *
 *   // Export a specific element
 *   await exportPageToPDF({
 *     element: document.getElementById('main-content'),
 *     title: 'Dashboard Report',
 *     filename: 'dashboard-report',
 *   });
 *
 *   // Export with button state management
 *   btn.addEventListener('click', () => exportPageToPDF({
 *     button: btn,
 *     element: document.querySelector('main'),
 *     title: 'Risk Report',
 *     filename: 'risk-report',
 *   }));
 *
 * @module pdf-export
 */

let _libsLoaded = false;

/**
 * Lazy-load jsPDF and html2canvas from CDN.
 */
async function _loadLibs() {
  if (_libsLoaded) return;

  const load = (url) => new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(s);
  });

  await load('https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js');
  await load('https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js');
  _libsLoaded = true;
}

/**
 * Convert any Plotly charts in the element to static PNG images.
 * This is needed because html2canvas cannot capture WebGL/SVG Plotly charts.
 *
 * @param {HTMLElement} container - Container to process
 * @returns {Function} Cleanup function to restore original Plotly divs
 */
async function _convertPlotlyCharts(container) {
  const plotlyDivs = container.querySelectorAll('.js-plotly-plot, [class*="plotly"]');
  const restoreFns = [];

  for (const div of plotlyDivs) {
    try {
      if (typeof Plotly !== 'undefined' && div.data) {
        const imgData = await Plotly.toImage(div, {
          format: 'png',
          width: 800,
          height: 400,
        });
        const img = document.createElement('img');
        img.src = imgData;
        img.style.width = '100%';
        img.style.maxWidth = '800px';

        const originalHTML = div.innerHTML;
        const originalStyle = div.style.cssText;
        div.innerHTML = '';
        div.appendChild(img);

        restoreFns.push(() => {
          div.innerHTML = originalHTML;
          div.style.cssText = originalStyle;
        });
      }
    } catch (e) {
      console.warn('Plotly chart conversion failed:', e);
    }
  }

  return () => restoreFns.forEach(fn => fn());
}

/**
 * Add header and footer to PDF pages.
 *
 * @param {jsPDF} pdf - jsPDF instance
 * @param {string} title - Report title
 * @param {number} pageCount - Total pages
 */
function _addHeaderFooter(pdf, title, pageCount) {
  const pages = pdf.getNumberOfPages();
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();

  for (let i = 1; i <= pages; i++) {
    pdf.setPage(i);

    // Header
    pdf.setFontSize(8);
    pdf.setTextColor(100, 100, 100);
    pdf.text(`SmartFolio — ${title}`, 10, 8);
    pdf.text(new Date().toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    }), pageWidth - 10, 8, { align: 'right' });

    // Header line
    pdf.setDrawColor(200, 200, 200);
    pdf.line(10, 10, pageWidth - 10, 10);

    // Footer
    pdf.setFontSize(7);
    pdf.setTextColor(150, 150, 150);
    pdf.text(`Page ${i} of ${pages}`, pageWidth / 2, pageHeight - 5, { align: 'center' });
  }
}

/**
 * Export a page element to PDF.
 *
 * @param {Object} options
 * @param {HTMLElement} options.element - DOM element to capture
 * @param {string} [options.title='SmartFolio Report'] - Report title (shown in header)
 * @param {string} [options.filename='smartfolio-report'] - Filename (without .pdf)
 * @param {HTMLButtonElement} [options.button] - Button to show loading state on
 * @param {string} [options.orientation='portrait'] - 'portrait' or 'landscape'
 * @param {number} [options.scale=2] - Render scale (2 = retina quality)
 * @param {string[]} [options.hideSelectors=[]] - CSS selectors to hide during capture
 */
export async function exportPageToPDF(options = {}) {
  const {
    element,
    title = 'SmartFolio Report',
    filename = 'smartfolio-report',
    button = null,
    orientation = 'portrait',
    scale = 2,
    hideSelectors = ['.pdf-hide', '.icon-btn', '.refresh-btn', 'nav', 'domain-nav', '.page-header button'],
  } = options;

  if (!element) {
    console.error('pdf-export: no element provided');
    return;
  }

  let originalBtnHTML = '';
  if (button) {
    originalBtnHTML = button.innerHTML;
    button.innerHTML = 'Loading libs...';
    button.disabled = true;
  }

  try {
    // 1. Load libraries
    await _loadLibs();

    if (button) button.innerHTML = 'Generating PDF...';

    // 2. Hide elements we don't want in the PDF
    const hidden = [];
    for (const sel of hideSelectors) {
      element.querySelectorAll(sel).forEach(el => {
        if (el.style.display !== 'none') {
          hidden.push({ el, prev: el.style.display });
          el.style.display = 'none';
        }
      });
    }

    // 3. Convert Plotly charts to images
    const restorePlotly = await _convertPlotlyCharts(element);

    // 4. Wait for any pending renders
    await new Promise(r => setTimeout(r, 300));

    // 5. Capture with html2canvas
    const canvas = await html2canvas(element, {
      scale,
      backgroundColor: '#ffffff',
      useCORS: true,
      logging: false,
      // Force light theme colors for PDF
      onclone: (doc) => {
        doc.documentElement.setAttribute('data-theme', 'light');
      },
    });

    // 6. Restore hidden elements and Plotly charts
    hidden.forEach(({ el, prev }) => { el.style.display = prev; });
    restorePlotly();

    // 7. Create PDF with multi-page support
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation, unit: 'mm', format: 'a4' });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 10;
    const usableWidth = pageWidth - (margin * 2);
    const usableHeight = pageHeight - 15 - margin; // 15mm top for header

    const imgWidth = usableWidth;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    if (imgHeight <= usableHeight) {
      // Single page
      pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, 15, imgWidth, imgHeight);
    } else {
      // Multi-page: slice canvas
      const totalPages = Math.ceil(imgHeight / usableHeight);
      const sliceHeight = Math.floor(canvas.height / totalPages);

      for (let page = 0; page < totalPages; page++) {
        if (page > 0) pdf.addPage();

        const srcY = page * sliceHeight;
        const srcH = Math.min(sliceHeight, canvas.height - srcY);

        // Create a slice canvas
        const sliceCanvas = document.createElement('canvas');
        sliceCanvas.width = canvas.width;
        sliceCanvas.height = srcH;
        const sliceCtx = sliceCanvas.getContext('2d');
        sliceCtx.drawImage(canvas, 0, srcY, canvas.width, srcH, 0, 0, canvas.width, srcH);

        const sliceImgHeight = (srcH * imgWidth) / canvas.width;
        pdf.addImage(sliceCanvas.toDataURL('image/png'), 'PNG', margin, 15, imgWidth, sliceImgHeight);
      }
    }

    // 8. Add headers/footers
    _addHeaderFooter(pdf, title, pdf.getNumberOfPages());

    // 9. Save
    const dateStr = new Date().toISOString().split('T')[0];
    pdf.save(`${filename}_${dateStr}.pdf`);

    if (button) {
      button.innerHTML = 'PDF saved!';
      setTimeout(() => {
        button.innerHTML = originalBtnHTML;
        button.disabled = false;
      }, 2000);
    }

  } catch (error) {
    console.error('PDF export failed:', error);
    if (button) {
      button.innerHTML = originalBtnHTML;
      button.disabled = false;
    }
    // Show user-friendly error
    if (window.showToast) {
      window.showToast('PDF export failed. Please try again.', 'error');
    }
  }
}

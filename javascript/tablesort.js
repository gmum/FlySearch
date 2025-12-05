// noinspection JSUnresolvedReference
document$.subscribe(function () {
    // noinspection JSUnresolvedReference
    Tablesort.extend('pmnumber', function (item) {
        // Test if item is a number (with or without ± std) or em dash
        // Matches: "45.5", "45.5 ± 3.5", "—"
        // Also handles &nbsp; (non-breaking space) and actual non-breaking space character (\u00a0)
        const normalized = item.replace(/&nbsp;/g, ' ').replace(/\u00a0/g, ' ');
        return /^[0-9.]+(\s*±\s*[0-9.]+)?$/.test(normalized) || item.trim() === '—' || item.trim() === '&mdash;';
    }, function (a, b) {
        // Extract the main number from the format "NUMBER ± STD" or just "NUMBER"
        // Treat "—" as negative infinity
        function extractNumber(item) {
            // Replace &nbsp; and non-breaking space character with regular space
            const normalized = item.replace(/&nbsp;/g, ' ').replace(/\u00a0/g, ' ');
            const trimmed = normalized.trim();

            // Check if it's the em dash (missing value) - also check for &mdash; entity
            if (trimmed === '—' || trimmed === '&mdash;') {
                return -Infinity;
            }

            // Extract the first number from patterns like "45.5 ± 3.5" or "45.5"
            const match = trimmed.match(/^([0-9.]+)/);
            if (match) {
                return parseFloat(match[1]);
            }

            // Fallback to negative infinity if parsing fails
            return -Infinity;
        }

        const numA = extractNumber(a);
        const numB = extractNumber(b);

        return numA - numB;
    });
    const tables = document.querySelectorAll("article table:not([class])");
    tables.forEach(function (table) {
        // noinspection JSUnresolvedReference
        new Tablesort(table)
    })
})

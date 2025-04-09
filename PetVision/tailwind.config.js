/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pets/templates/**/*.html', // Include all HTML files in the templates directory
    './pets/static/**/*.js',      // Include any JavaScript files in the static directory
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};


/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*",  
  "./static/src/**/*.js",
  "./node_modules/flowbite/**/*.js"],
  theme: {
    extend: {},
  },
  plugins: [
    require("flowbite/plugin")
  ],
}


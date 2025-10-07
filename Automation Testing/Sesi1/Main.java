package seleniumSesi3;

import org.openqa.selenium.By;
import org.openqa.selenium.By.ByXPath;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

public class Main {

	public static void main(String[] args) {
//		// Set Property
//		System.setProperty("webdriver.com.driver", "D:/chromedriver-win64/chromedriver-win64/chromedriver.exe");
//		// ganti slash nya jadi back slash
//		
//		// 2. initialize webDriver
//		WebDriver diver = new ChromeDriver();
//		
//		// open page, google
//		diver.get("https://google.com");
//		diver.manage().window().maximize();
//		
//		// ambil title dan url
//		// ambil suatu data dari page
//		String title = diver.getTitle();
//		System.out.println("Judul Page : " + title);
//		
//		String currentURL = diver.getCurrentUrl();
//		System.out.println("Current URL : " + currentURL);
//		
//		// navigation
//		// 1. pindah ke web lab
//		diver.navigate().to("https://x.com/");
//		// 2. balik ke google
//		diver.navigate().back();
//		// 3. balik ke wab lab lagi
//		diver.navigate().forward();
//		// 4. refresh page
//		diver.navigate().refresh();
//		// 5. quit chrome driver
//		diver.quit();
		
		// login suatu web page
		// initialize
		WebDriver driver = new ChromeDriver();
		
		// 1. open page
		driver.get("https://saucedemo.com");
		
		// 2. locate username & password
		WebElement usernameField = driver.findElement(By.id("user-name"));
		WebElement passwordField = driver.findElement(By.name("password"));
		
		// 3. input username & password
		usernameField.sendKeys("standard_user");
		passwordField.sendKeys("secret_sauce");
		
		// 4. locate login button
		WebElement loginbuButton = driver.findElement(By.xpath("//*[@id=\"login-button\"]"));
		
		// 5. click login button
		loginbuButton.click();
		
		// element states (isDisplayed, isEnabled, isSelected)
		WebElement cart = driver.findElement(By.id("shopping_cart_container"));
		
		if(cart.isDisplayed()) {
			System.out.println("Login berhasil - cart tampil");
		}
		else { 
			System.out.println("Login gagal");
		}

	}

}

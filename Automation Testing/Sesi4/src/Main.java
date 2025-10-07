import java.time.Duration;


import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class Main {
	public static void main(String[] args) {
//		System.setProperty("webdriver.chrome.driver", "D:/chromedriver-win64/chromedriver-win64/chromedriver.exe");
		
		WebDriver driver = new ChromeDriver();
		
		driver.get("https://saucedemo.com");
		
		driver.manage().window().maximize();
//		driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(3));
		
		// Explicit Wait
		
		WebDriverWait wait5sec  = new WebDriverWait(driver, Duration.ofSeconds(5));
		WebDriverWait wait10sec = new WebDriverWait(driver, Duration.ofSeconds(10));
		
		WebElement username = driver.findElement(By.xpath("//*[@id=\"user-name\"]"));
		WebElement password = driver.findElement(By.xpath("//*[@id=\"password\"]"));
		WebElement login = driver.findElement(By.xpath("//*[@id=\"login-button\"]"));
		
		username.sendKeys("locked_out_user");
		password.sendKeys("secret_sauce");
		login.click();
		
		try {
			WebElement logoCart = wait5sec.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//*[@id=\"header_container\"]/div[1]/div[2]/div")));
			
			System.out.println("Login Berhasil");
			
		} catch (NoSuchElementException e) {
			// TODO: handle exception
			System.out.println("Login Gagal");
		} catch (TimeoutException e) {
			// TODO: handle exception
			System.out.println("Error timeout");
		}
		 
		
	
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		driver.quit();
		
	}
}

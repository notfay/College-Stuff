import java.time.Duration;
import java.util.NoSuchElementException;

import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class Main {
	public static void main(String[] args) {
		//System.setProperty()
		
				WebDriver driver = new ChromeDriver();
				driver.get("https://www.saucedemo.com/");
				
				driver.manage().window().maximize();
//				driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(2));
//				
				WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(2));
				WebDriverWait wait2 = new WebDriverWait(driver, Duration.ofSeconds(5));
				
						
				WebElement username = driver.findElement(By.xpath("//*[@id=\"user-name\"]"));
				username.sendKeys("standard_user");
				
				WebElement password = driver.findElement(By.xpath("//*[@id=\"password\"]"));
				password.sendKeys("secret_sauce");
				
				WebElement login = driver.findElement(By.xpath("//*[@id=\"login-button\"]"));
				login.click();
				try {
					//WebElement title = wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//*[@id=\\\"user-name\\\"]")));
					//WebElement title = wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//*[@id=\"header_container\"]/div[1]/div[2]/div")));
					
					System.out.println("Berhasil login");
					
				} catch (NoSuchElementException e) {
					System.out.println("Gagal Login");
					// TODO: handle exception
				} catch (TimeoutException e) {
					System.out.println("Error timeout");
				}
				
				System.out.println("Silahkan Masuk");
				
				
				try {
					Thread.sleep(30000);
				} catch (Exception e) {
					e.printStackTrace();
				}
			
				driver.quit();
	}
}
